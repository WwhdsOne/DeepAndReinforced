"""训练过程可视化 —— 读取 history.json 生成独立 HTML 仪表盘。"""

from __future__ import annotations

import json
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MNIST 训练过程</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js">
</script>
<style>
  :root {
    --bg: #0f1117;
    --card: #161822;
    --border: #252836;
    --text: #c9d1d9;
    --muted: #6e7681;
    --accent: #58a6ff;
    --green: #3fb950;
    --orange: #d2991d;
    --red: #f85149;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 24px; font-weight: 600; margin-bottom: 8px; }
  .subtitle { color: var(--muted); font-size: 14px; margin-bottom: 32px; }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
  }
  .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .stat-value { font-size: 28px; font-weight: 700; }
  .stat-value.accent { color: var(--accent); }
  .stat-value.green { color: var(--green); }
  .stat-value.orange { color: var(--orange); }

  .chart-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 24px;
  }
  .chart-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; }
  .chart-wrap { position: relative; width: 100%; }
  .chart-wrap canvas { width: 100% !important; }

  .params-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }
  .params-table th {
    text-align: left;
    padding: 10px 16px;
    color: var(--muted);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
  }
  .params-table td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
  }
  .params-table code {
    background: var(--bg);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 13px;
    color: var(--accent);
  }
</style>
</head>
<body>
<div class="container">
  <h1>MNIST 多层感知机 — 训练过程</h1>
  <p class="subtitle">全批量梯度下降，无正则化，Xavier 初始化</p>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">最终损失</div>
      <div class="stat-value orange">{{final_loss}}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">测试准确率</div>
      <div class="stat-value green">{{final_accuracy}}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">训练轮数</div>
      <div class="stat-value accent">{{epochs}}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">隐藏层大小</div>
      <div class="stat-value">{{hidden_size}}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">学习率</div>
      <div class="stat-value">{{learning_rate}}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">训练集样本</div>
      <div class="stat-value">{{limit_train}}</div>
    </div>
  </div>

  <div class="chart-section">
    <div class="chart-title">训练损失曲线</div>
    <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
  </div>

  <div class="chart-section">
    <div class="chart-title">测试准确率曲线</div>
    <div class="chart-wrap"><canvas id="accuracyChart"></canvas></div>
  </div>

  <div class="chart-section">
    <div class="chart-title">超参数配置</div>
    <table class="params-table">
      <tr><th>参数</th><th>值</th></tr>
      <tr><td>网络结构</td><td><code>{{layers}}</code></td></tr>
      <tr><td>隐藏层大小</td><td><code>{{hidden_size}}</code></td></tr>
      <tr><td>学习率</td><td><code>{{learning_rate}}</code></td></tr>
      <tr><td>训练轮数</td><td><code>{{epochs}}</code></td></tr>
      <tr><td>训练集样本数</td><td><code>{{limit_train}}</code></td></tr>
      <tr><td>测试集样本数</td><td><code>{{limit_test}}</code></td></tr>
    </table>
  </div>
</div>

<script>
const lossData = {{loss_json}};
const accuracyData = {{accuracy_json}};
const epochs = lossData.length;
const labels = Array.from({length: epochs}, (_, i) => i + 1);

const darkOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
  },
  scales: {
    x: {
      title: { display: true, text: 'Epoch', color: '#6e7681' },
      ticks: { color: '#6e7681' },
      grid: { color: '#252836' },
    },
  },
};

// Loss chart
new Chart(document.getElementById('lossChart'), {
  type: 'line',
  data: {
    labels: labels,
    datasets: [{
      data: lossData,
      borderColor: '#d2991d',
      backgroundColor: 'rgba(210,153,29,0.08)',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: '#d2991d',
      fill: true,
      tension: 0.3,
    }],
  },
  options: {
    ...darkOptions,
    plugins: {
      ...darkOptions.plugins,
      tooltip: {
        callbacks: {
          label: ctx => `Loss: ${ctx.raw.toFixed(6)}`,
        },
      },
    },
    scales: {
      ...darkOptions.scales,
      y: {
        title: { display: true, text: 'Cross-Entropy Loss', color: '#6e7681' },
        ticks: { color: '#6e7681', callback: v => v.toFixed(4) },
        grid: { color: '#252836' },
      },
    },
  },
});

// Accuracy chart
const borderColor = accuracyData[accuracyData.length - 1] >= 0.95 ? '#3fb950' : '#d2991d';
new Chart(document.getElementById('accuracyChart'), {
  type: 'line',
  data: {
    labels: labels,
    datasets: [{
      data: accuracyData.map(v => v * 100),
      borderColor: borderColor,
      backgroundColor: 'rgba(63,185,80,0.08)',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: borderColor,
      fill: true,
      tension: 0.3,
    }],
  },
  options: {
    ...darkOptions,
    plugins: {
      ...darkOptions.plugins,
      tooltip: {
        callbacks: {
          label: ctx => `Accuracy: ${ctx.raw.toFixed(2)}%`,
        },
      },
    },
    scales: {
      ...darkOptions.scales,
      y: {
        title: { display: true, text: 'Accuracy (%)', color: '#6e7681' },
        ticks: { color: '#6e7681', callback: v => v.toFixed(1) + '%' },
        grid: { color: '#252836' },
        min: Math.max(0, Math.min(...accuracyData) * 100 - 5),
        max: 100,
      },
    },
  },
});
</script>
</body>
</html>"""


def generate_html(history_path: str | Path, output_path: str | Path | None = None) -> Path:
    """从 history.json 生成可视化 HTML。"""
    history_path = Path(history_path)
    with open(history_path) as f:
        data = json.load(f)

    hp = data["hyperparameters"]
    loss = data["loss"]
    accuracy = data.get("accuracy", [])

    if output_path is None:
        output_path = history_path.with_suffix(".html")
    else:
        output_path = Path(output_path)

    html = (
        HTML_TEMPLATE
        .replace("{{final_loss}}", f"{data['final_loss']:.6f}")
        .replace("{{final_accuracy}}", f"{data['final_accuracy'] * 100:.2f}%")
        .replace("{{epochs}}", str(hp["epochs"]))
        .replace("{{hidden_size}}", str(hp["hidden_size"]))
        .replace("{{learning_rate}}", str(hp["learning_rate"]))
        .replace("{{limit_train}}", str(hp["limit_train"]))
        .replace("{{limit_test}}", str(hp["limit_test"]))
        .replace("{{layers}}", " → ".join(str(l) for l in hp["layers"]))
        .replace("{{loss_json}}", json.dumps(loss))
        .replace("{{accuracy_json}}", json.dumps(accuracy))
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


def visualize_main() -> None:
    """CLI 入口：mnist-visualize。"""
    import argparse

    _pkg_root = Path(__file__).resolve().parent.parent.parent  # experiments/neural_network/

    parser = argparse.ArgumentParser(description="生成 MNIST 训练过程可视化 HTML")
    parser.add_argument(
        "history",
        nargs="?",
        type=Path,
        default=_pkg_root / "artifacts" / "mnist_mlp.history.json",
        help="训练历史 JSON 文件路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 HTML 文件路径（默认与 history 同名 .html）",
    )
    args = parser.parse_args()

    if not args.history.exists():
        print(f"错误：找不到训练历史文件 {args.history}")
        print("请先运行 mnist-train 训练模型。")
        return

    out = generate_html(args.history, args.output)
    print(f"可视化页面已生成：{out}")
