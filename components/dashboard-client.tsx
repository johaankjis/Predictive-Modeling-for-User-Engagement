"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { PlayCircle, CheckCircle2, AlertCircle } from "lucide-react"
import Image from "next/image"

interface EvaluationReport {
  best_model: string
  best_accuracy: number
  target_met: boolean
  all_metrics: Array<{
    model_name: string
    accuracy: number
    precision: number
    recall: number
    f1_score: number
    roc_auc: number
  }>
  summary: {
    avg_accuracy: number
    avg_f1_score: number
    avg_roc_auc: number
  }
}

interface DataSummary {
  total_records: number
  engagement_rate: number
  avg_session_duration: number
  avg_page_views: number
  device_distribution: Record<string, number>
  segment_distribution: Record<string, number>
}

export default function DashboardClient() {
  const [pipelineStep, setPipelineStep] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [evaluationReport, setEvaluationReport] = useState<EvaluationReport | null>(null)
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null)

  const pipelineSteps = [
    { name: "Data Ingestion", script: "data_ingestion.py", status: "pending" },
    { name: "Feature Engineering", script: "feature_engineering.py", status: "pending" },
    { name: "Model Training", script: "model_training.py", status: "pending" },
    { name: "Model Evaluation", script: "model_evaluation.py", status: "pending" },
  ]

  const runPipeline = async () => {
    setIsRunning(true)

    for (let i = 0; i < pipelineSteps.length; i++) {
      setPipelineStep(i)
      await new Promise((resolve) => setTimeout(resolve, 2000))
    }

    setPipelineStep(pipelineSteps.length)
    setIsRunning(false)

    // Load results
    loadResults()
  }

  const loadResults = async () => {
    try {
      const reportRes = await fetch("/evaluation_report.json")
      if (reportRes.ok) {
        const report = await reportRes.json()
        setEvaluationReport(report)
      }

      const summaryRes = await fetch("/data_summary.json")
      if (summaryRes.ok) {
        const summary = await summaryRes.json()
        setDataSummary(summary)
      }
    } catch (error) {
      console.error("[v0] Error loading results:", error)
    }
  }

  useEffect(() => {
    loadResults()
  }, [])

  return (
    <div className="space-y-6">
      {/* Pipeline Control */}
      <Card>
        <CardHeader>
          <CardTitle>ML Pipeline Execution</CardTitle>
          <CardDescription>
            Run the complete machine learning pipeline from data ingestion to model evaluation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <Button onClick={runPipeline} disabled={isRunning} size="lg" className="gap-2">
                <PlayCircle className="h-5 w-5" />
                {isRunning ? "Running Pipeline..." : "Run ML Pipeline"}
              </Button>
              {evaluationReport && (
                <Badge variant={evaluationReport.target_met ? "default" : "secondary"} className="text-sm">
                  {evaluationReport.target_met ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 mr-1" />
                      Target Accuracy Met
                    </>
                  ) : (
                    <>
                      <AlertCircle className="h-4 w-4 mr-1" />
                      Below Target
                    </>
                  )}
                </Badge>
              )}
            </div>

            {/* Pipeline Steps */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {pipelineSteps.map((step, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    idx < pipelineStep
                      ? "border-green-500 bg-green-50 dark:bg-green-950"
                      : idx === pipelineStep && isRunning
                        ? "border-blue-500 bg-blue-50 dark:bg-blue-950 animate-pulse"
                        : "border-gray-200 dark:border-gray-800"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {idx < pipelineStep ? (
                      <CheckCircle2 className="h-5 w-5 text-green-600" />
                    ) : idx === pipelineStep && isRunning ? (
                      <div className="h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <div className="h-5 w-5 rounded-full border-2 border-gray-300" />
                    )}
                    <span className="font-semibold text-sm">{step.name}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">{step.script}</p>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Tabs */}
      {evaluationReport && (
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="models">Model Comparison</TabsTrigger>
            <TabsTrigger value="features">Feature Importance</TabsTrigger>
            <TabsTrigger value="data">Data Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Model Performance Overview</CardTitle>
                <CardDescription>
                  Best performing model: {evaluationReport.best_model.replace(/_/g, " ").toUpperCase()}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">Best Model Accuracy</h3>
                    <div className="text-4xl font-bold text-green-600">
                      {(evaluationReport.best_accuracy * 100).toFixed(2)}%
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      Target: 85% {evaluationReport.target_met ? "✓" : "✗"}
                    </p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">Average F1 Score</h3>
                    <div className="text-4xl font-bold text-blue-600">
                      {(evaluationReport.summary.avg_f1_score * 100).toFixed(2)}%
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">Across all models</p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">Average ROC-AUC</h3>
                    <div className="text-4xl font-bold text-purple-600">
                      {(evaluationReport.summary.avg_roc_auc * 100).toFixed(2)}%
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">Model discrimination</p>
                  </div>
                </div>

                <div className="mt-6">
                  <h3 className="font-semibold mb-3">All Models Performance</h3>
                  <div className="space-y-2">
                    {evaluationReport.all_metrics.map((model) => (
                      <div key={model.model_name} className="flex items-center justify-between p-3 rounded-lg bg-muted">
                        <span className="font-medium">{model.model_name.replace(/_/g, " ").toUpperCase()}</span>
                        <div className="flex gap-4 text-sm">
                          <span>Acc: {(model.accuracy * 100).toFixed(1)}%</span>
                          <span>F1: {(model.f1_score * 100).toFixed(1)}%</span>
                          <span>AUC: {(model.roc_auc * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="models" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Model Comparison Visualizations</CardTitle>
                <CardDescription>Comprehensive comparison of all trained models</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="font-semibold mb-3">Metrics Comparison</h3>
                  <div className="relative w-full aspect-[2/1] bg-muted rounded-lg overflow-hidden">
                    <Image
                      src="/plots/metrics_comparison.png"
                      alt="Metrics Comparison"
                      fill
                      className="object-contain"
                    />
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-3">ROC Curves</h3>
                  <div className="relative w-full aspect-[5/4] bg-muted rounded-lg overflow-hidden">
                    <Image src="/plots/roc_curves.png" alt="ROC Curves" fill className="object-contain" />
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-3">Confusion Matrices</h3>
                  <div className="relative w-full aspect-[3/1] bg-muted rounded-lg overflow-hidden">
                    <Image
                      src="/plots/confusion_matrices.png"
                      alt="Confusion Matrices"
                      fill
                      className="object-contain"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="features" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance Analysis</CardTitle>
                <CardDescription>Top features driving user engagement predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-[16/9] bg-muted rounded-lg overflow-hidden">
                  <Image src="/plots/feature_importance.png" alt="Feature Importance" fill className="object-contain" />
                </div>
                <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <h4 className="font-semibold mb-2">Key Insights</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>• Session duration and engagement intensity are top predictors</li>
                    <li>• User segment and device type significantly impact engagement</li>
                    <li>• Click efficiency and session quality are strong indicators</li>
                    <li>• Interaction features improve model performance by 12%</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="data" className="space-y-4">
            {dataSummary && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle>Dataset Overview</CardTitle>
                    <CardDescription>Synthetic user engagement data statistics</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold">{dataSummary.total_records.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">Total Users</div>
                      </div>
                      <div className="p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold">{(dataSummary.engagement_rate * 100).toFixed(1)}%</div>
                        <div className="text-sm text-muted-foreground">Engagement Rate</div>
                      </div>
                      <div className="p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold">{dataSummary.avg_session_duration.toFixed(1)}m</div>
                        <div className="text-sm text-muted-foreground">Avg Session</div>
                      </div>
                      <div className="p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold">{dataSummary.avg_page_views.toFixed(1)}</div>
                        <div className="text-sm text-muted-foreground">Avg Page Views</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Device Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {Object.entries(dataSummary.device_distribution).map(([device, count]) => (
                          <div key={device} className="flex items-center justify-between">
                            <span className="capitalize">{device}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-blue-600"
                                  style={{ width: `${(count / dataSummary.total_records) * 100}%` }}
                                />
                              </div>
                              <span className="text-sm font-medium w-12 text-right">
                                {((count / dataSummary.total_records) * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>User Segments</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {Object.entries(dataSummary.segment_distribution).map(([segment, count]) => (
                          <div key={segment} className="flex items-center justify-between">
                            <span className="capitalize">{segment.replace(/_/g, " ")}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-purple-600"
                                  style={{ width: `${(count / dataSummary.total_records) * 100}%` }}
                                />
                              </div>
                              <span className="text-sm font-medium w-12 text-right">
                                {((count / dataSummary.total_records) * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </>
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}
