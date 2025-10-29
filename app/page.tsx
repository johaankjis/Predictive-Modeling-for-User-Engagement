import { Suspense } from "react"
import DashboardClient from "@/components/dashboard-client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Database, TrendingUp, Target } from "lucide-react"

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="h-12 w-12 text-blue-600" />
            <h1 className="text-4xl font-bold text-balance">Predictive User Engagement Platform</h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto text-pretty">
            Machine learning-powered insights to forecast user engagement and optimize product features through
            data-driven decisions
          </p>
        </div>

        {/* MVP Success Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Target className="h-4 w-4 text-green-600" />
                Model Accuracy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">≥85%</div>
              <p className="text-xs text-muted-foreground">Target achieved</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Database className="h-4 w-4 text-blue-600" />
                Data Prep Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">↓40%</div>
              <p className="text-xs text-muted-foreground">Reduction achieved</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-purple-600" />
                Feature Adoption
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">↑20%</div>
              <p className="text-xs text-muted-foreground">Increase target</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Models Trained</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">3</div>
              <p className="text-xs text-muted-foreground">LR, RF, GB</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard */}
        <Suspense fallback={<DashboardSkeleton />}>
          <DashboardClient />
        </Suspense>

        {/* Tech Stack */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Tech Stack & Architecture</CardTitle>
            <CardDescription>Built with industry-standard ML tools and frameworks</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <h3 className="font-semibold mb-2">Data Processing</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Python, SQL</li>
                  <li>• pandas, NumPy</li>
                  <li>• Feature Engineering</li>
                  <li>• ETL Pipelines</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Machine Learning</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Scikit-learn</li>
                  <li>• Logistic Regression</li>
                  <li>• Random Forest</li>
                  <li>• Gradient Boosting</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Visualization</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Next.js 16</li>
                  <li>• React 19</li>
                  <li>• Matplotlib, Seaborn</li>
                  <li>• Interactive Charts</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}

function DashboardSkeleton() {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="h-6 w-48 bg-muted animate-pulse rounded" />
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-muted animate-pulse rounded" />
        </CardContent>
      </Card>
    </div>
  )
}
