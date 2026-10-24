param(
    [string]$MetricsPath = "$PSScriptRoot/../outputs/tinystories-evaluation-v1/metrics.csv",
    [string]$ImagePath = "$PSScriptRoot/../experiments/tinystories-evaluation-loss.png"
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $ImagePath) { throw 'Output image already exists; choose a new path.' }
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
$evaluationRows = Import-Csv -LiteralPath $MetricsPath
$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
try {
    $chart.Width = 1000
    $chart.Height = 600
    $chart.BackColor = [System.Drawing.Color]::White
    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea
    $area.AxisX.Title = 'Completed training steps'
    $area.AxisX.Minimum = 250
    $area.AxisX.Maximum = 1000
    $area.AxisX.Interval = 250
    $area.AxisY.Title = 'Mean token cross-entropy (nats)'
    $area.AxisY.Minimum = 3.1
    $area.AxisY.Maximum = 4.1
    $area.AxisY.Interval = 0.2
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $chart.ChartAreas.Add($area)
    $chart.Titles.Add('TinyStories: fixed-window train / validation evaluation') | Out-Null
    $chart.Titles.Add('3 fixed seeds; 98,304 sampled targets per split per checkpoint') | Out-Null
    $chart.Legends.Add('Legend') | Out-Null
    foreach ($split in @('train', 'val')) {
        $series = $chart.Series.Add($split)
        $series.ChartType = 'Line'
        $series.BorderWidth = 3
        $series.MarkerStyle = 'Circle'
        $series.MarkerSize = 8
        $series.Color = if ($split -eq 'train') { [System.Drawing.Color]::RoyalBlue } else { [System.Drawing.Color]::DarkOrange }
        foreach ($row in $evaluationRows) {
            $loss = [double]::Parse($row.($split + '_loss'), [System.Globalization.CultureInfo]::InvariantCulture)
            $series.Points.AddXY([int]$row.step, $loss) | Out-Null
        }
    }
    $chart.SaveImage([System.IO.Path]::GetFullPath($ImagePath), 'Png')
} finally {
    $chart.Dispose()
}
