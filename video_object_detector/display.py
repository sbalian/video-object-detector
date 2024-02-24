import rich.console
import rich.progress

CONSOLE = rich.console.Console()

STATUS = CONSOLE.status
PRINT = CONSOLE.print

PROGRESS = rich.progress.Progress(
    rich.progress.TextColumn("[progress.description]{task.description}"),
    rich.progress.BarColumn(),
    rich.progress.TaskProgressColumn(),
)
