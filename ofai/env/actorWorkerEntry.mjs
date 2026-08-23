/**
 * Node worker entry that registers tsx before loading the TypeScript actor.
 * worker_threads does not apply --import tsx to the worker graph reliably.
 */
import { register } from "tsx/esm/api";

register();
await import("./actorWorker.ts");
