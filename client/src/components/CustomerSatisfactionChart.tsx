import React from "react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";
import { Card, CardBody, CardHeader } from "@nextui-org/react";
import { IconMoodHappy } from "@tabler/icons-react";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";

interface SatisfactionData {
  satisfaction: string;
  value: number;
  fill: string;
}

interface Props {
  sentimentData: Record<string, number>;
}

const chartConfig = {
  "highly positive": {
    label: "Highly Positive",
    color: "#17C964",
  },
  positive: {
    label: "Positive",
    color: "#4CC38A",
  },
  neutral: {
    label: "Neutral",
    color: "#889096",
  },
  negative: {
    label: "Negative",
    color: "#F5A524",
  },
  "highly negative": {
    label: "Highly Negative",
    color: "#F31260",
  },
};

export const CustomerSatisfactionChart: React.FC<Props> = ({
  sentimentData,
}) => {
  const data = Object.entries(sentimentData).map(([satisfaction, count]) => ({
    satisfaction: satisfaction
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" "),
    value: count,
    fill: chartConfig[satisfaction.toLowerCase() as keyof typeof chartConfig]
      .color,
  }));

  const totalResponses = React.useMemo(() => {
    return data.reduce((acc, curr) => acc + curr.value, 0);
  }, [data]);

  return (
    <Card className="w-full mb-6">
      <CardHeader>
        <IconMoodHappy className="mr-2 text-primary" size={24} />
        <h2 className="text-xl font-semibold">Customer Satisfaction</h2>
      </CardHeader>
      <CardBody>
        <ChartContainer config={chartConfig} className="pt-2 h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={data} cx="50%" cy="50%">
              <PolarGrid />
              <PolarAngleAxis
                dataKey="satisfaction"
                tick={{ fill: "currentcolor" }}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Radar
                name="Incidents"
                dataKey="value"
                stroke="hsl(var(--chart-1))"
                fill="hsl(var(--chart-1))"
                fillOpacity={0.2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardBody>
    </Card>
  );
};
