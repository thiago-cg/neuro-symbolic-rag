import pino from "pino";

let _logger: pino.Logger | undefined;

export function getLogger(name?: string): pino.Logger {
  if (!_logger) {
    const isDev = process.env["NODE_ENV"] !== "production";
    _logger = pino({
      level: process.env["LOG_LEVEL"] ?? "info",
      ...(isDev
        ? {
            transport: {
              target: "pino-pretty",
              options: { colorize: true, translateTime: "SYS:standard" },
            },
          }
        : {}),
    });
  }
  return name ? _logger.child({ module: name }) : _logger;
}

/** Wraps an async function and logs duration + result status */
export function withTiming<T extends unknown[], R>(
  name: string,
  fn: (...args: T) => Promise<R>,
): (...args: T) => Promise<R> {
  const log = getLogger("timing");
  return async (...args: T): Promise<R> => {
    const start = Date.now();
    try {
      const result = await fn(...args);
      log.info({ step: name, durationMs: Date.now() - start }, `${name} ok`);
      return result;
    } catch (err) {
      log.error(
        { step: name, durationMs: Date.now() - start, err },
        `${name} failed`,
      );
      throw err;
    }
  };
}
