"use client";

import { useState } from "react";
import {
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Card, CardHeader, CardBody, CardFooter } from "@nextui-org/card";
import { Select, SelectSection, SelectItem } from "@nextui-org/select";
import { RadioGroup, Radio } from "@nextui-org/radio";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";
import { IconGraph, IconCalendarClock } from "@tabler/icons-react";

type TimeUnit = "hour" | "day" | "month";
type DisplayOption = "opened" | "closed" | "both";

interface IncidentTimingChartProps {
  timingData: {
    hourly: Array<{ time: string; opened: number; closed: number }>;
    daily: Array<{ time: string; opened: number; closed: number }>;
    monthly: Array<{ time: string; opened: number; closed: number }>;
  };
}

const chartConfig = {
  opened: {
    label: "Incidents Opened",
    color: "var(--chart-opened)",
  },
  closed: {
    label: "Incidents Closed",
    color: "var(--chart-closed)",
  },
};

export function IncidentTimingChart({ timingData }: IncidentTimingChartProps) {
  const [timeUnit, setTimeUnit] = useState<TimeUnit>("hour");
  const [displayOption, setDisplayOption] = useState<DisplayOption>("both");

  const getChartData = () => {
    switch (timeUnit) {
      case "hour":
        return timingData.hourly;
      case "day":
        return timingData.daily;
      case "month":
        return timingData.monthly;
      default:
        return [];
    }
  };

  const chartData = getChartData();

  return (
    <Card className="w-[50vw] mb-6">
      <CardHeader>
        <IconGraph className="mr-2 text-primary" size={24} />
        <h2 className="text-xl font-semibold">Incident Timing</h2>
      </CardHeader>
      <CardBody>
        <div className="flex flex-col sm:flex-row gap-4 mb-2">
          <div className="flex-1">
            <Select
              value={timeUnit}
              onChange={(e) => setTimeUnit(e.target.value as TimeUnit)}
              defaultSelectedKeys={["hour"]}
              startContent={<IconCalendarClock size={24} />}
            >
              <SelectItem key="hour" value="hour">
                Hours of the Day
              </SelectItem>
              <SelectItem key="day" value="day">
                Days of the Week
              </SelectItem>
              <SelectItem key="month" value="month">
                Months of the Year
              </SelectItem>
            </Select>
          </div>
          <div className="flex-1">
            <RadioGroup
              value={displayOption}
              orientation="horizontal"
              defaultValue="both"
              onValueChange={(value) =>
                setDisplayOption(value as DisplayOption)
              }
              className="flex gap-4"
            >
              <Radio key="opened" value="opened">
                Opened
              </Radio>
              <Radio key="closed" value="closed">
                Closed
              </Radio>
              <Radio key="both" value="both">
                Both
              </Radio>
            </RadioGroup>
          </div>
        </div>
        <div className=" pt-2">
          <ChartContainer config={chartConfig}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="time" tickLine={false} axisLine={false} />
                <YAxis
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => Math.floor(value).toString()}
                  domain={[0, "auto"]}
                  allowDataOverflow={true}
                  interval="preserveStartEnd"
                  tickCount={5}
                />
                {(displayOption === "opened" || displayOption === "both") && (
                  <Line
                    type="monotone"
                    dataKey="opened"
                    stroke={chartConfig.opened.color}
                    strokeWidth={2}
                    dot={false}
                  />
                )}
                {(displayOption === "closed" || displayOption === "both") && (
                  <Line
                    type="monotone"
                    dataKey="closed"
                    stroke={chartConfig.closed.color}
                    strokeWidth={2}
                    dot={false}
                  />
                )}
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </CardBody>
    </Card>
  );
}
