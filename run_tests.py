#!/usr/bin/env python3
"""
VPS-FastSearch Phase 2 Test Suite

Runs comprehensive tests for daemon mode and generates benchmark report.
"""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add fastsearch to path
sys.path.insert(0, str(Path(__file__).parent))

import psutil


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    mode: str
    passed: bool
    time_ms: float
    memory_before_mb: float = 0
    memory_after_mb: float = 0
    notes: str = ""
    command: str = ""


@dataclass
class TestSuite:
    """Collection of test results."""
    results: list[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    
    def add(self, result: TestResult):
        self.results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: {result.time_ms:.1f}ms - {result.notes}")
    
    def summary(self) -> dict:
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{100*passed/len(self.results):.1f}%" if self.results else "N/A",
        }


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_command(cmd: str, timeout: float = 120) -> tuple[str, float, bool]:
    """Run command and return (output, time_ms, success)."""
    # Replace 'fastsearch' with python module call
    if cmd.startswith("fastsearch "):
        cmd = "python3 -m fastsearch.cli " + cmd[11:]
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(__file__).parent)
        )
        elapsed = (time.perf_counter() - start) * 1000
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return output, elapsed, success
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout * 1000, False
    except Exception as e:
        return str(e), 0, False


def check_daemon_running() -> bool:
    """Check if daemon is running."""
    output, _, _ = run_command("fastsearch daemon status --json")
    try:
        data = json.loads(output)
        return data.get("running", False) is not False
    except:
        return False


def stop_daemon():
    """Stop daemon if running."""
    if check_daemon_running():
        run_command("fastsearch daemon stop")
        time.sleep(1)


def start_daemon() -> float:
    """Start daemon and return startup time in ms."""
    stop_daemon()
    start = time.perf_counter()
    
    # Start in background
    subprocess.Popen(
        ["python3", "-m", "fastsearch.cli", "daemon", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).parent),
    )
    
    # Wait for daemon to be ready
    for _ in range(60):  # 60 seconds max
        time.sleep(0.5)
        if check_daemon_running():
            break
    
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed


def get_daemon_status() -> dict | None:
    """Get daemon status."""
    output, _, success = run_command("fastsearch daemon status --json")
    if not success:
        return None
    try:
        return json.loads(output)
    except:
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("VPS-FastSearch Phase 2 Test Suite")
    print("=" * 60)
    print()
    
    suite = TestSuite()
    os.chdir(Path(__file__).parent)
    
    # Ensure we have test data
    db_path = "fastsearch.db"
    if not Path(db_path).exists():
        print("Creating test database...")
        run_command("fastsearch index README.md")
    
    # =========================================================================
    # Test 1: Cold Start (Daemon Start)
    # =========================================================================
    print("\n[Test 1] Cold Start - Daemon Start")
    print("-" * 40)
    
    stop_daemon()
    mem_before = get_memory_mb()
    startup_time = start_daemon()
    
    status = get_daemon_status()
    mem_after = status.get("total_memory_mb", 0) if status else 0
    embedder_loaded = "embedder" in status.get("loaded_models", {}) if status else False
    
    suite.add(TestResult(
        name="Cold Start",
        mode="daemon start",
        passed=embedder_loaded,
        time_ms=startup_time,
        memory_before_mb=mem_before,
        memory_after_mb=mem_after,
        notes=f"Embedder loaded: {embedder_loaded}, Memory: {mem_after:.0f}MB",
        command="fastsearch daemon start",
    ))
    
    # =========================================================================
    # Test 2: Warm Search (via socket)
    # =========================================================================
    print("\n[Test 2] Warm Search - Via Socket")
    print("-" * 40)
    
    # First search to ensure warm
    run_command('fastsearch search "test query" --json')
    
    # Measure warm search
    output, time_ms, success = run_command('fastsearch search "configuration settings" --json')
    
    try:
        data = json.loads(output)
        search_time = data.get("search_time_ms", time_ms)
        result_count = len(data.get("results", []))
        used_daemon = data.get("daemon", False)
    except:
        search_time = time_ms
        result_count = 0
        used_daemon = False
    
    suite.add(TestResult(
        name="Warm Search",
        mode="via socket",
        passed=success and used_daemon and search_time < 50,
        time_ms=search_time,
        notes=f"Results: {result_count}, Daemon: {used_daemon}",
        command='fastsearch search "configuration settings"',
    ))
    
    # =========================================================================
    # Test 3: Direct Search (--no-daemon)
    # =========================================================================
    print("\n[Test 3] Direct Search - No Daemon")
    print("-" * 40)
    
    output, time_ms, success = run_command('fastsearch search "configuration" --no-daemon --json')
    
    try:
        data = json.loads(output)
        search_time = data.get("search_time_ms", time_ms)
        used_daemon = data.get("daemon", False)
    except:
        search_time = time_ms
        used_daemon = True  # Should be False
    
    suite.add(TestResult(
        name="Direct Search",
        mode="--no-daemon",
        passed=success and not used_daemon,
        time_ms=search_time,
        notes=f"Daemon bypassed: {not used_daemon}",
        command='fastsearch search "configuration" --no-daemon',
    ))
    
    # =========================================================================
    # Test 4: Rerank On-Demand (Cold)
    # =========================================================================
    print("\n[Test 4] Rerank On-Demand - Cold")
    print("-" * 40)
    
    # Unload reranker first via client
    try:
        from vps_fastsearch import FastSearchClient
        client = FastSearchClient()
        client.unload_model("reranker")
        client.close()
    except:
        pass
    
    time.sleep(1)
    
    output, time_ms, success = run_command('fastsearch search "test query" --rerank --json')
    
    try:
        data = json.loads(output)
        search_time = data.get("search_time_ms", time_ms)
        reranked = data.get("reranked", False)
    except:
        search_time = time_ms
        reranked = False
    
    suite.add(TestResult(
        name="Rerank Cold",
        mode="--rerank (cold)",
        passed=success and reranked,
        time_ms=search_time,
        notes=f"Includes model load time",
        command='fastsearch search "test query" --rerank',
    ))
    
    # =========================================================================
    # Test 5: Rerank Warm
    # =========================================================================
    print("\n[Test 5] Rerank - Warm")
    print("-" * 40)
    
    # Reranker should now be loaded
    output, time_ms, success = run_command('fastsearch search "memory management" --rerank --json')
    
    try:
        data = json.loads(output)
        search_time = data.get("search_time_ms", time_ms)
        reranked = data.get("reranked", False)
    except:
        search_time = time_ms
        reranked = False
    
    suite.add(TestResult(
        name="Rerank Warm",
        mode="--rerank (hot)",
        passed=success and reranked and search_time < 500,  # Cross-encoder takes ~200ms
        time_ms=search_time,
        notes=f"Model already loaded",
        command='fastsearch search "memory management" --rerank',
    ))
    
    # =========================================================================
    # Test 6: Daemon Status
    # =========================================================================
    print("\n[Test 6] Daemon Status")
    print("-" * 40)
    
    output, time_ms, success = run_command("fastsearch daemon status --json")
    
    try:
        data = json.loads(output)
        has_models = "loaded_models" in data
        has_memory = "total_memory_mb" in data
        has_uptime = "uptime_seconds" in data
    except:
        has_models = has_memory = has_uptime = False
    
    suite.add(TestResult(
        name="Daemon Status",
        mode="status command",
        passed=success and has_models and has_memory and has_uptime,
        time_ms=time_ms,
        notes=f"Has models: {has_models}, memory: {has_memory}, uptime: {has_uptime}",
        command="fastsearch daemon status",
    ))
    
    # =========================================================================
    # Test 7: Python Client
    # =========================================================================
    print("\n[Test 7] Python Client")
    print("-" * 40)
    
    try:
        from vps_fastsearch import FastSearchClient
        
        start = time.perf_counter()
        client = FastSearchClient()
        result = client.search("daemon mode")
        client.close()
        elapsed = (time.perf_counter() - start) * 1000
        
        success = len(result.get("results", [])) > 0
        notes = f"Results: {len(result.get('results', []))}"
    except Exception as e:
        success = False
        elapsed = 0
        notes = str(e)
    
    suite.add(TestResult(
        name="Python Client",
        mode="client library",
        passed=success,
        time_ms=elapsed,
        notes=notes,
        command="FastSearchClient().search('query')",
    ))
    
    # =========================================================================
    # Test 8: Client Embed
    # =========================================================================
    print("\n[Test 8] Client Embed")
    print("-" * 40)
    
    try:
        from vps_fastsearch import FastSearchClient
        
        start = time.perf_counter()
        client = FastSearchClient()
        result = client.embed(["test text 1", "test text 2"])
        client.close()
        elapsed = (time.perf_counter() - start) * 1000
        
        embeddings = result.get("embeddings", [])
        success = len(embeddings) == 2 and len(embeddings[0]) == 768
        notes = f"Embeddings: {len(embeddings)}, dims: {len(embeddings[0]) if embeddings else 0}"
    except Exception as e:
        success = False
        elapsed = 0
        notes = str(e)
    
    suite.add(TestResult(
        name="Client Embed",
        mode="embed API",
        passed=success,
        time_ms=elapsed,
        notes=notes,
        command="FastSearchClient().embed(['text'])",
    ))
    
    # =========================================================================
    # Test 9: Config Reload
    # =========================================================================
    print("\n[Test 9] Config Reload")
    print("-" * 40)
    
    output, time_ms, success = run_command("fastsearch daemon reload")
    
    suite.add(TestResult(
        name="Config Reload",
        mode="reload command",
        passed=success and "reloaded" in output.lower(),
        time_ms=time_ms,
        notes="Config reloaded without restart" if success else output[:50],
        command="fastsearch daemon reload",
    ))
    
    # =========================================================================
    # Test 10: Daemon Stop
    # =========================================================================
    print("\n[Test 10] Daemon Stop")
    print("-" * 40)
    
    output, time_ms, success = run_command("fastsearch daemon stop")
    time.sleep(1)
    
    still_running = check_daemon_running()
    
    suite.add(TestResult(
        name="Daemon Stop",
        mode="stop command",
        passed=success and not still_running,
        time_ms=time_ms,
        notes=f"Clean shutdown: {not still_running}",
        command="fastsearch daemon stop",
    ))
    
    # =========================================================================
    # Summary
    # =========================================================================
    suite.end_time = datetime.now()
    summary = suite.summary()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total:   {summary['total']}")
    print(f"Passed:  {summary['passed']}")
    print(f"Failed:  {summary['failed']}")
    print(f"Rate:    {summary['pass_rate']}")
    print()
    
    # Generate HTML report
    generate_report(suite)
    
    return 0 if summary['failed'] == 0 else 1


def generate_report(suite: TestSuite):
    """Generate HTML benchmark report."""
    
    # Calculate performance comparisons
    warm_search = next((r for r in suite.results if r.name == "Warm Search"), None)
    direct_search = next((r for r in suite.results if r.name == "Direct Search"), None)
    rerank_cold = next((r for r in suite.results if r.name == "Rerank Cold"), None)
    rerank_warm = next((r for r in suite.results if r.name == "Rerank Warm"), None)
    cold_start = next((r for r in suite.results if r.name == "Cold Start"), None)
    
    speedup = (direct_search.time_ms / warm_search.time_ms) if warm_search and direct_search and warm_search.time_ms > 0 else 0
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>VPS-FastSearch Daemon Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .card h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .card .value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .card.pass .value {{ color: #4CAF50; }}
        .card.fail .value {{ color: #f44336; }}
        .card.speed .value {{ color: #2196F3; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f9f9f9; }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .fail {{ color: #f44336; font-weight: bold; }}
        .time {{ font-family: monospace; }}
        .comparison {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .bar-container {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .bar-label {{ width: 120px; font-weight: bold; }}
        .bar {{
            height: 24px;
            background: #4CAF50;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            color: white;
            font-size: 12px;
        }}
        .bar.slow {{ background: #ff9800; }}
        .bar.direct {{ background: #2196F3; }}
        .recommendation {{
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>🚀 VPS-FastSearch Daemon Benchmark Report</h1>
    <p>Generated: {suite.end_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <div class="card pass">
            <h3>Tests Passed</h3>
            <div class="value">{suite.summary()['passed']}/{suite.summary()['total']}</div>
        </div>
        <div class="card speed">
            <h3>Daemon Speedup</h3>
            <div class="value">{speedup:.0f}x</div>
        </div>
        <div class="card">
            <h3>Cold Start</h3>
            <div class="value">{cold_start.time_ms/1000:.1f}s</div>
        </div>
        <div class="card">
            <h3>Warm Search</h3>
            <div class="value">{warm_search.time_ms:.0f}ms</div>
        </div>
    </div>
    
    <h2>📊 Test Results</h2>
    <table>
        <thead>
            <tr>
                <th>Test</th>
                <th>Mode</th>
                <th>Time</th>
                <th>Result</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
'''
    
    for r in suite.results:
        status_class = "pass" if r.passed else "fail"
        status_text = "✓ PASS" if r.passed else "✗ FAIL"
        html += f'''            <tr>
                <td>{r.name}</td>
                <td>{r.mode}</td>
                <td class="time">{r.time_ms:.1f}ms</td>
                <td class="{status_class}">{status_text}</td>
                <td>{r.notes}</td>
            </tr>
'''
    
    html += '''        </tbody>
    </table>
    
    <h2>⚡ Speed Comparison</h2>
    <div class="comparison">
        <h3>Search Latency (lower is better)</h3>
'''
    
    # Calculate bar widths
    max_time = max(
        warm_search.time_ms if warm_search else 0,
        direct_search.time_ms if direct_search else 0,
        rerank_warm.time_ms if rerank_warm else 0,
    ) or 100
    
    if warm_search:
        width = (warm_search.time_ms / max_time) * 100
        html += f'''        <div class="bar-container">
            <div class="bar-label">Daemon (warm)</div>
            <div class="bar" style="width: {max(width, 5)}%">{warm_search.time_ms:.0f}ms</div>
        </div>
'''
    
    if direct_search:
        width = (direct_search.time_ms / max_time) * 100
        html += f'''        <div class="bar-container">
            <div class="bar-label">Direct (cold)</div>
            <div class="bar direct" style="width: {max(width, 5)}%">{direct_search.time_ms:.0f}ms</div>
        </div>
'''
    
    if rerank_warm:
        width = (rerank_warm.time_ms / max_time) * 100
        html += f'''        <div class="bar-container">
            <div class="bar-label">Rerank (warm)</div>
            <div class="bar slow" style="width: {max(width, 5)}%">{rerank_warm.time_ms:.0f}ms</div>
        </div>
'''
    
    html += '''    </div>
    
    <h2>💾 Memory Usage</h2>
    <div class="comparison">
'''
    
    if cold_start:
        html += f'''        <p><strong>After daemon start:</strong> {cold_start.memory_after_mb:.0f}MB (embedder loaded)</p>
'''
    
    html += '''    </div>
    
    <h2>💡 Recommendations</h2>
    <div class="recommendation">
        <strong>For production use:</strong>
        <ul>
            <li>Always run the daemon for interactive applications</li>
            <li>Use <code>--detach</code> or systemd for background operation</li>
            <li>Configure idle timeout for reranker to save memory</li>
            <li>Monitor memory usage with <code>fastsearch daemon status</code></li>
        </ul>
    </div>
    
    <footer>
        VPS-FastSearch v0.2.0 | Benchmark run on {suite.start_time.strftime("%Y-%m-%d")}
    </footer>
</body>
</html>
'''
    
    # Write report to project directory
    report_path = Path(__file__).parent / "fastsearch_daemon_report.html"
    report_path.write_text(html)
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
