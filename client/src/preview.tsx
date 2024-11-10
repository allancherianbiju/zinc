import React, { useMemo, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Card,
  CardBody,
  Button,
  Progress,
  Calendar,
  Chip,
} from "@nextui-org/react";
import { IconTable, IconColumns, IconUser } from "@tabler/icons-react";
import Navbar from "./components/navbar";

import {
  Table,
  TableHeader,
  TableBody,
  TableColumn,
  TableRow,
  TableCell,
} from "@nextui-org/react";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Key } from "@react-types/shared";
import { ScrollShadow } from "@nextui-org/scroll-shadow";
const MAX_TEXT_LENGTH = 50; // Maximum length for truncated text

const truncateText = (text: string, maxLength: number) => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + "...";
};

const calculateEmptyPercentage = (
  data: Record<string, any>[],
  column: string
): string => {
  const emptyCount = data.filter((row) => !row[column]).length;
  return ((emptyCount / data.length) * 100).toFixed(2);
};

const calculateColumnEmptyPercentage = (
  data: Record<string, any>[],
  column: string
): string => {
  const totalValues = data.length;
  const emptyCount = data.filter((row) => {
    const value = row[column];
    return value === null || value === undefined || value === "";
  }).length;

  return ((emptyCount / totalValues) * 100).toFixed(2);
};

const chartConfig = {
  desktop: {
    label: "Desktop",
    color: "#2563eb",
  },
  mobile: {
    label: "Mobile",
    color: "#60a5fa",
  },
} satisfies ChartConfig;

// Add these types at the top of the file
type SortDescriptor = {
  column?: Key;
  direction?: "ascending" | "descending";
};

// Add this constant at the top with other constants
type DataType = "unknown" | "string" | "number" | "date" | "boolean";

const DATA_TYPE_ORDER: Record<DataType, number> = {
  unknown: 0,
  string: 1,
  number: 2,
  date: 3,
  boolean: 4,
};

export default function Preview() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const location = useLocation();
  const navigate = useNavigate();
  const { fileData } = location.state || {};
  const [sortDescriptor, setSortDescriptor] = React.useState<SortDescriptor>(
    {}
  );

  if (!fileData) {
    navigate("/");
    return null;
  }

  const columns = useMemo(() => {
    if (fileData.preview_data && fileData.preview_data.length > 0) {
      return Object.keys(fileData.preview_data[0]);
    }
    return [];
  }, [fileData.preview_data]);

  const mappingData = {
    columns: columns,
    // Include any other necessary data
  };

  const sampleValues = columns.map((column) => {
    const samples = fileData.preview_data
      .slice(0, 3)
      .map((row: Record<string, any>) =>
        truncateText(String(row[column]), MAX_TEXT_LENGTH)
      )
      .join(", ");
    return samples || "N/A";
  });

  // Add this sorting function
  const sortedItems = React.useMemo(() => {
    let sortedData = [...columns];

    if (sortDescriptor.column && sortDescriptor.direction) {
      sortedData = sortedData.sort((a, b) => {
        let first = a;
        let second = b;
        let multiplier = sortDescriptor.direction === "ascending" ? 1 : -1;

        switch (sortDescriptor.column) {
          case "Empty Percentage":
            first = calculateColumnEmptyPercentage(
              fileData.preview_data,
              a
            ).toString();
            second = calculateColumnEmptyPercentage(
              fileData.preview_data,
              b
            ).toString();
            break;
          case "Data Type":
            const typeA = inferDataType(
              fileData.preview_data,
              a
            ).toLowerCase() as DataType;
            const typeB = inferDataType(
              fileData.preview_data,
              b
            ).toLowerCase() as DataType;
            return (
              (DATA_TYPE_ORDER[typeA] - DATA_TYPE_ORDER[typeB]) * multiplier
            );
          case "Column Name":
            return a.localeCompare(b) * multiplier;
          default:
            break;
        }

        return (first < second ? -1 : first > second ? 1 : 0) * multiplier;
      });
    }

    return sortedData;
  }, [columns, sortDescriptor, fileData.preview_data]);

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-4">Dataset Preview</h1>

        <Card className="mb-6">
          <CardBody>
            {/* <h2 className="text-xl font-semibold mb-4">General Statistics</h2> */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-1">
              <div className="flex items-center">
                <IconUser className="text-primary mr-4" size={60} />
                <div>
                  <h3 className="text-lg font-semibold">Account</h3>
                  <p className="text-2xl font-bold text-primary">
                    {fileData.account || "N/A"}
                  </p>
                </div>
              </div>
              <div className="flex items-center">
                <IconTable className="text-primary mr-4" size={60} />
                <div>
                  <h3 className="text-lg font-semibold">Rows</h3>
                  <p className="text-2xl font-bold text-primary">
                    {fileData.rows}
                  </p>
                </div>
              </div>
              <div className="flex items-center">
                <IconColumns className="text-primary mr-4" size={60} />
                <div>
                  <h3 className="text-lg font-semibold">Columns</h3>
                  <p className="text-2xl font-bold text-primary">
                    {fileData.columns}
                  </p>
                </div>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card className="mb-6">
          <CardBody>
            {/* <h2 className="text-xl font-semibold mb-4">Column Details</h2> */}
            <Table
              isHeaderSticky
              aria-label="Column details table"
              sortDescriptor={sortDescriptor}
              onSortChange={(descriptor) => setSortDescriptor(descriptor)}
              classNames={{
                base: "max-h-[570px]",
              }}
            >
              <TableHeader>
                <TableColumn key="Column Name" allowsSorting>
                  COLUMN NAME
                </TableColumn>
                <TableColumn key="Data Type" allowsSorting>
                  DATA TYPE
                </TableColumn>
                <TableColumn key="Empty Percentage" allowsSorting>
                  EMPTY PERCENTAGE
                </TableColumn>
                <TableColumn key="Sample Values" allowsSorting>
                  SAMPLE VALUES
                </TableColumn>
              </TableHeader>
              <TableBody>
                {sortedItems.map((column, index) => {
                  const emptyPercentage = calculateColumnEmptyPercentage(
                    fileData.preview_data,
                    column
                  );
                  const dataType = inferDataType(fileData.preview_data, column);

                  return (
                    <TableRow key={column}>
                      <TableCell>{column}</TableCell>
                      <TableCell>
                        <Chip
                          variant="flat"
                          color={getChipColorForDataType(dataType)}
                        >
                          {capitalizeFirstLetter(dataType)}
                        </Chip>
                      </TableCell>
                      <TableCell>{emptyPercentage}%</TableCell>
                      <TableCell>{sampleValues[index]}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardBody>
        </Card>

        <div className="mt-6 flex justify-end">
          <Button
            color="primary"
            onClick={() => {
              navigate("/mapping", { state: { fileData: mappingData } });
            }}
          >
            Proceed to Mapping
          </Button>
        </div>
      </main>
    </div>
  );
}

function inferDataType(data: Record<string, any>[], column: string): string {
  const nonEmptyValues = data.map((row) => row[column]).filter(Boolean);
  if (nonEmptyValues.length === 0) return "unknown";

  const sample = nonEmptyValues[0];
  if (typeof sample === "number") return "number";
  if (typeof sample === "boolean") return "boolean";
  if (!isNaN(Date.parse(sample))) return "date";
  return "string";
}

function capitalizeFirstLetter(string: string): string {
  return string.charAt(0).toUpperCase() + string.slice(1).toLowerCase();
}

function getChipColorForDataType(
  dataType: string
): "default" | "primary" | "secondary" | "success" | "warning" {
  switch (dataType.toLowerCase()) {
    case "number":
      return "primary";
    case "string":
      return "success";
    case "date":
      return "secondary";
    case "boolean":
      return "warning";
    default:
      return "default";
  }
}
