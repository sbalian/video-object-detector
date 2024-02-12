import rich.console
import rich.progress

CONSOLE = rich.console.Console()

status = CONSOLE.status
print = CONSOLE.print

PROGRESS = rich.progress.Progress(
    rich.progress.TextColumn("[progress.description]{task.description}"),
    rich.progress.BarColumn(),
    rich.progress.TaskProgressColumn(),
)
