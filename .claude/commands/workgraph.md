# WorkGraph

If `$ARGUMENTS` is empty, `start`, or `listen`, run `workgraph start claude`. If `$ARGUMENTS` looks like a filesystem path, run `workgraph start "$ARGUMENTS" claude`. If it is `stop`, run `workgraph stop .`. If it is `status`, run `workgraph status .`. If it starts with `reuse `, run `workgraph reuse` with the remaining text. If it starts with `export `, run `workgraph skill export` with the remaining text as the skill name and query. Then keep the answer short. If WorkGraph refuses to start because Claude is running from the home folder, tell the user to run `/workgraph /absolute/project/path`.

Keep the answer short. If the command fails because WorkGraph is not installed on PATH, ask the user to run `npm link` or install the package globally.
