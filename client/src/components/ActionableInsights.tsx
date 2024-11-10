import React from "react";
import { Card, CardBody, CardHeader } from "@nextui-org/react";
import {
  IconBulb,
  IconTrendingUp,
  IconAlertTriangle,
} from "@tabler/icons-react";

interface InsightData {
  category: string;
  metric: string;
  value: number | string;
  trend: "up" | "down" | "stable";
  impact: "high" | "medium" | "low";
  recommendation: string;
}

interface Props {
  insights: InsightData[];
}

export const ActionableInsights: React.FC<Props> = ({ insights }) => {
  return (
    <Card className="w-full">
      <CardHeader className="flex gap-3">
        <IconBulb className="text-primary" size={24} />
        <div className="flex flex-col">
          <p className="text-xl font-semibold">Actionable Insights</p>
          <p className="text-small text-default-500">
            Key opportunities to improve your Overall Performance Score
          </p>
        </div>
      </CardHeader>
      <CardBody>
        <div className="space-y-4">
          {insights.map((insight, index) => (
            <div key={index} className="border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                {insight.impact === "high" && (
                  <IconAlertTriangle className="text-danger" size={20} />
                )}
                <h3 className="font-medium">{insight.category}</h3>
              </div>
              <p className="text-default-500 mb-2">
                {insight.metric}:{" "}
                <span className="font-semibold">{insight.value}</span>
                <IconTrendingUp
                  className={`inline ml-2 ${
                    insight.trend === "up" ? "text-success" : "text-danger"
                  }`}
                  size={16}
                />
              </p>
              <p className="text-sm">{insight.recommendation}</p>
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  );
};
