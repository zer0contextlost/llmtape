import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
import yaml

console = Console()


@click.group()
def cli():
    """llmtape -- record and replay LLM API calls in tests"""


@cli.command("list")
@click.option("--cassette-dir", default=".cassettes", show_default=True)
def list_cassettes(cassette_dir):
    """List all cassettes with age, function, provider, and token counts."""
    from ._cassette import age_days

    d = Path(cassette_dir)
    if not d.exists():
        console.print(f"[yellow]No cassette directory found: {cassette_dir}[/yellow]")
        return

    files = sorted(d.glob("*.yaml"))
    if not files:
        console.print("[yellow]No cassettes found.[/yellow]")
        return

    table = Table("File", "Function", "Provider", "Model", "Age (days)", "Tokens")
    for f in files:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            meta = data.get("metadata", {})
            req = data.get("request", {}).get("normalized", {})
            usage = data.get("response", {}).get("raw", {}).get("usage", {})
            tokens = usage.get("total_tokens") or (
                (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
            )
            days = f"{age_days(f):.1f}"
            table.add_row(
                f.name,
                meta.get("function_name", "-"),
                data.get("provider", "-"),
                req.get("model", "-"),
                days,
                str(tokens) if tokens else "-",
            )
        except Exception as e:
            table.add_row(f.name, "[red]parse error[/red]", "-", "-", "-", str(e)[:40])

    console.print(table)


@cli.command()
@click.argument("name")
@click.option("--cassette-dir", default=".cassettes", show_default=True)
def show(name, cassette_dir):
    """Pretty-print a cassette."""
    path = Path(cassette_dir) / name
    if not path.exists():
        path = Path(cassette_dir) / f"{name}.yaml"
    if not path.exists():
        console.print(f"[red]Cassette not found: {name}[/red]")
        return

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    req = data.get("request", {}).get("normalized", {})
    raw = data.get("response", {}).get("raw", {})

    console.print(f"\n[bold]{path.name}[/bold]")
    console.print(f"  Provider:    {data.get('provider', '-')}")
    console.print(f"  Function:    {meta.get('function_name', '-')}")
    console.print(f"  Recorded:    {meta.get('recorded_at', '-')}")
    console.print(f"  SDK:         {meta.get('sdk_version', '-')}")
    console.print(f"  Latency:     {meta.get('latency_ms', '-')}ms")
    console.print(f"  Model:       {req.get('model', '-')}")

    messages = req.get("messages", [])
    if messages:
        console.print("\n[bold]Messages:[/bold]")
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str):
                preview = content[:200] + ("..." if len(content) > 200 else "")
                console.print(f"  [{role}] {preview}")

    # Extract response content
    provider = data.get("provider", "")
    if provider == "openai":
        choices = raw.get("choices", [{}])
        msg = choices[0].get("message", {}) if choices else {}
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        usage = raw.get("usage", {})
        console.print(f"\n[bold]Response:[/bold]")
        if content:
            preview = content[:300] + ("..." if len(content) > 300 else "")
            console.print(f"  {preview}")
        if tool_calls:
            console.print(f"  [yellow]tool_calls: {len(tool_calls)} call(s)[/yellow]")
        console.print(f"\n  Tokens: {usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out")
    elif provider == "anthropic":
        content_blocks = raw.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text":
                text = block.get("text", "")
                preview = text[:300] + ("..." if len(text) > 300 else "")
                console.print(f"\n[bold]Response:[/bold]")
                console.print(f"  {preview}")
        usage = raw.get("usage", {})
        console.print(f"\n  Tokens: {usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out")


@cli.command()
@click.option("--cassette-dir", default=".cassettes", show_default=True)
@click.option("--max-age", default=30, show_default=True, help="Flag cassettes older than N days")
def check(cassette_dir, max_age):
    """Flag stale cassettes older than --max-age days."""
    from ._cassette import age_days

    d = Path(cassette_dir)
    if not d.exists():
        console.print(f"[yellow]No cassette directory: {cassette_dir}[/yellow]")
        return

    files = sorted(d.glob("*.yaml"))
    stale = [(f, age_days(f)) for f in files if age_days(f) > max_age]

    if not stale:
        console.print(f"[green]All cassettes are fresh (within {max_age} days)[/green]")
        return

    console.print(f"[red]{len(stale)} stale cassette(s) (older than {max_age} days):[/red]\n")
    for f, days in stale:
        console.print(f"  {f.name}  --  {days:.0f} days old")
    console.print(f"\nRe-record with: LLMTAPE_MODE=record-missing pytest")


@cli.command()
@click.argument("pattern")
@click.option("--cassette-dir", default=".cassettes", show_default=True)
@click.option("--yes", is_flag=True, help="Skip confirmation")
def delete(pattern, cassette_dir, yes):
    """Delete cassette(s) by name or glob pattern."""
    d = Path(cassette_dir)
    matches = list(d.glob(pattern if "*" in pattern else f"*{pattern}*"))
    matches = [f for f in matches if f.suffix == ".yaml"]

    if not matches:
        console.print(f"[yellow]No cassettes match: {pattern}[/yellow]")
        return

    console.print(f"Found {len(matches)} cassette(s):")
    for f in matches:
        console.print(f"  {f.name}")

    if not yes:
        click.confirm("\nDelete these cassettes?", abort=True)

    for f in matches:
        f.unlink()
    console.print(f"[green]Deleted {len(matches)} cassette(s)[/green]")
