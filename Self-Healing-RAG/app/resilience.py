import time
import random
import logging
import threading
from typing import Any, ClassVar, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

logger = logging.getLogger("SelfHealingRAG.Resilience")


class TokenBucketRateLimiter:
    """Thread-safe Token Bucket Rate Limiter to throttle Groq API requests."""
    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # Tokens added per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            while True:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                    
                # Calculate necessary sleep duration
                needed = 1.0 - self.tokens
                wait_time = needed / self.rate
                time.sleep(wait_time)


class CircuitBreakerOpenException(Exception):
    """Raised when the circuit breaker is in the OPEN state and blocks requests."""
    pass


class CircuitBreaker:
    """Thread-safe Circuit Breaker pattern implementation."""
    def __init__(self, failure_threshold: int = 3, recovery_time: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.failures = 0
        self.last_state_change = time.time()
        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        with self.lock:
            now = time.time()
            if self.state == "OPEN":
                if now - self.last_state_change > self.recovery_time:
                    self.state = "HALF-OPEN"
                    self.last_state_change = now
                    logger.info("Circuit Breaker transitioned to HALF-OPEN. Running test request.")
                else:
                    raise CircuitBreakerOpenException("Circuit breaker is OPEN. Call blocked.")

        try:
            result = func(*args, **kwargs)
            with self.lock:
                if self.state == "HALF-OPEN" or self.failures > 0:
                    logger.info(f"Circuit Breaker call succeeded. Transitioning from {self.state} to CLOSED.")
                    self.state = "CLOSED"
                    self.failures = 0
                    self.last_state_change = time.time()
            return result
        except Exception as e:
            with self.lock:
                self.failures += 1
                logger.warning(f"Circuit Breaker failure recorded ({self.failures}/{self.failure_threshold}): {e}")
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    self.last_state_change = time.time()
                    logger.critical(f"Circuit Breaker tripped to OPEN for {self.recovery_time} seconds.")
            raise e


class RetryBudget:
    """Tracks LLM request budget to prevent retry storms."""
    def __init__(self, budget_ratio: float = 0.2, min_requests: int = 5):
        self.budget_ratio = budget_ratio
        self.min_requests = min_requests
        self.success_count = 0
        self.retry_count = 0
        self.lock = threading.Lock()

    def record_success(self) -> None:
        with self.lock:
            self.success_count += 1

    def record_retry(self) -> bool:
        with self.lock:
            total = self.success_count + self.retry_count
            if total < self.min_requests:
                self.retry_count += 1
                return True
            allowed_retries = int(self.success_count * self.budget_ratio)
            if self.retry_count < allowed_retries:
                self.retry_count += 1
                return True
            logger.warning(
                f"Retry budget exceeded (successes={self.success_count}, retries={self.retry_count}). "
                f"Blocking retry request."
            )
            return False


class ResilientChatGroq(ChatGroq):
    """
    Resilient wrapper for ChatGroq that subclasses the LangChain ChatGroq model.
    Enforces Rate Limiting, Circuit Breaker, and Retry Budget check on every call.
    """
    # Shared class-level instances for centralized tracking
    rate_limiter: ClassVar[TokenBucketRateLimiter] = TokenBucketRateLimiter(rate=0.5, capacity=2.0)  # 30 RPM limit, burst capacity of 2
    circuit_breaker: ClassVar[CircuitBreaker] = CircuitBreaker(failure_threshold=3, recovery_time=10.0)
    retry_budget: ClassVar[RetryBudget] = RetryBudget(budget_ratio=0.2, min_requests=5)

    def __init__(self, *args, **kwargs):
        # Enforce request timeout default of 10 seconds if not specified
        if "request_timeout" not in kwargs:
            kwargs["request_timeout"] = 10.0
        super().__init__(*args, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 1. Acquire Token from Rate Limiter (Queue / Throttling)
        self.rate_limiter.acquire()

        max_retries = 3
        base_delay = 2.0
        max_delay = 10.0
        jitter = 1.0

        for attempt in range(max_retries + 1):
            try:
                # 2. Invoke through Circuit Breaker
                res = self.circuit_breaker.call(
                    super()._generate, messages, stop=stop, run_manager=run_manager, **kwargs
                )
                self.retry_budget.record_success()
                return res
            except CircuitBreakerOpenException as cbe:
                raise cbe
            except Exception as exc:
                err_msg = str(exc)
                is_rate_limit = (
                    "429" in err_msg 
                    or "RateLimit" in err_msg 
                    or "rate limit" in err_msg.lower() 
                    or "resource_exhausted" in err_msg.lower()
                )
                is_timeout = (
                    "timeout" in err_msg.lower()
                    or "timed out" in err_msg.lower()
                    or "connecttimeout" in err_msg.lower()
                )
                
                # Check retry eligibility
                if (is_rate_limit or is_timeout) and attempt < max_retries:
                    if self.retry_budget.record_retry():
                        delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, jitter)
                        logger.warning(
                            f"Groq API call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                        continue
                logger.error(f"Permanent failure in ResilientChatGroq._generate: {exc}")
                raise exc

        raise RuntimeError("ResilientChatGroq failed to generate response after all retries.")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio
        # Async throttling
        await asyncio.to_thread(self.rate_limiter.acquire)

        max_retries = 3
        base_delay = 2.0
        max_delay = 10.0
        jitter = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Circuit Breaker check
                if self.circuit_breaker.state == "OPEN":
                    now = time.time()
                    if now - self.circuit_breaker.last_state_change > self.circuit_breaker.recovery_time:
                        self.circuit_breaker.state = "HALF-OPEN"
                        self.circuit_breaker.last_state_change = now
                        logger.info("Circuit Breaker transitioned to HALF-OPEN (async).")
                    else:
                        raise CircuitBreakerOpenException("Circuit breaker is OPEN. Call blocked.")

                res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                
                if self.circuit_breaker.state == "HALF-OPEN" or self.circuit_breaker.failures > 0:
                    self.circuit_breaker.state = "CLOSED"
                    self.circuit_breaker.failures = 0
                    self.circuit_breaker.last_state_change = time.time()
                
                self.retry_budget.record_success()
                return res
            except Exception as exc:
                if not isinstance(exc, CircuitBreakerOpenException):
                    self.circuit_breaker.failures += 1
                    if self.circuit_breaker.failures >= self.circuit_breaker.failure_threshold:
                        self.circuit_breaker.state = "OPEN"
                        self.circuit_breaker.last_state_change = time.time()
                        logger.critical(f"Circuit Breaker tripped to OPEN (async).")

                err_msg = str(exc)
                is_rate_limit = (
                    "429" in err_msg 
                    or "RateLimit" in err_msg 
                    or "rate limit" in err_msg.lower() 
                    or "resource_exhausted" in err_msg.lower()
                )
                is_timeout = (
                    "timeout" in err_msg.lower()
                    or "timed out" in err_msg.lower()
                    or "connecttimeout" in err_msg.lower()
                )
                
                if (is_rate_limit or is_timeout) and attempt < max_retries:
                    if self.retry_budget.record_retry():
                        delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, jitter)
                        logger.warning(
                            f"Groq API async call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                logger.error(f"Permanent failure in ResilientChatGroq._agenerate: {exc}")
                raise exc

        raise RuntimeError("ResilientChatGroq failed to generate response after all async retries.")
