import pRetry, { type Options } from "p-retry";
import { getLogger } from "./observability.js";

const log = getLogger("retry");

/** 3-attempt policy for LLM calls (exponential backoff 1s → 2s → 4s) */
export const llmRetryOptions: Options = {
  retries: 3,
  minTimeout: 1_000,
  maxTimeout: 8_000,
  factor: 2,
  onFailedAttempt: (err) => {
    log.warn(
      { attempt: err.attemptNumber, retriesLeft: err.retriesLeft },
      `LLM call failed, retrying…`,
    );
  },
};

/** 5-attempt policy for external API calls (longer backoff 2s → 4s → 8s) */
export const apiRetryOptions: Options = {
  retries: 5,
  minTimeout: 2_000,
  maxTimeout: 30_000,
  factor: 2,
  onFailedAttempt: (err) => {
    log.warn(
      { attempt: err.attemptNumber, retriesLeft: err.retriesLeft },
      `API call failed, retrying…`,
    );
  },
};

/** Convenience wrapper — run fn with LLM retry policy */
export function withLlmRetry<T>(fn: () => Promise<T>): Promise<T> {
  return pRetry(fn, llmRetryOptions);
}

/** Convenience wrapper — run fn with API retry policy */
export function withApiRetry<T>(fn: () => Promise<T>): Promise<T> {
  return pRetry(fn, apiRetryOptions);
}
