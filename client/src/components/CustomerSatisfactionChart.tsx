import React from "react";
import {
  PieChart,
  Pie,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  Label,
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
  count: number;
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
  const data: SatisfactionData[] = Object.entries(sentimentData).map(
    ([satisfaction, count]) => ({
      satisfaction: satisfaction
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" "),
      count,
      fill: chartConfig[satisfaction.toLowerCase() as keyof typeof chartConfig]
        .color,
    })
  );

  const totalResponses = React.useMemo(() => {
    return data.reduce((acc, curr) => acc + curr.count, 0);
  }, [data]);

  return (
    <Card className="w-full mb-6">
      <CardHeader>
        <IconMoodHappy className="mr-2 text-primary" size={24} />
        <h2 className="text-xl font-semibold">
          Customer Satisfaction Distribution
        </h2>
      </CardHeader>
      <CardBody>
        <ChartContainer config={chartConfig} className="pt-2 h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <ChartTooltip
                cursor={false}
                content={<ChartTooltipContent hideLabel />}
              />
              <Pie
                data={data}
                dataKey="count"
                nameKey="satisfaction"
                innerRadius={60}
                strokeWidth={5}
              >
                <Label
                  content={({ viewBox }) => {
                    if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                      return (
                        <text
                          x={viewBox.cx}
                          y={viewBox.cy}
                          textAnchor="middle"
                          dominantBaseline="middle"
                        >
                          <tspan
                            x={viewBox.cx}
                            y={viewBox.cy}
                            className="fill-foreground text-3xl font-bold"
                          >
                            {totalResponses.toLocaleString()}
                          </tspan>
                          <tspan
                            x={viewBox.cx}
                            y={(viewBox.cy || 0) + 24}
                            className="fill-muted-foreground"
                          >
                            Responses
                          </tspan>
                        </text>
                      );
                    }
                  }}
                />
              </Pie>
              <ChartLegend content={<ChartLegendContent />} />
            </PieChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardBody>
    </Card>
  );
};
