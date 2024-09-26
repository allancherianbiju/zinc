import React, { useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Breadcrumbs,
  BreadcrumbItem,
  Card,
  CardBody,
  Button,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Tooltip,
} from "@nextui-org/react";
import Navbar from "./components/navbar";

const MAX_TEXT_LENGTH = 50; // Maximum length for truncated text

const truncateText = (text: string, maxLength: number) => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + "...";
};

export default function Preview() {
  const location = useLocation();
  const navigate = useNavigate();
  const { fileData } = location.state || {};

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

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <Breadcrumbs className="mb-6">
          <BreadcrumbItem>Home</BreadcrumbItem>
          <BreadcrumbItem>Upload</BreadcrumbItem>
          <BreadcrumbItem>Preview</BreadcrumbItem>
          <BreadcrumbItem isDisabled>Map</BreadcrumbItem>
          <BreadcrumbItem isDisabled>Process</BreadcrumbItem>
          <BreadcrumbItem isDisabled>Report</BreadcrumbItem>
        </Breadcrumbs>

        <h1 className="text-2xl font-bold mb-4">Dataset Preview</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <Card>
            <CardBody>
              <h2 className="text-xl font-semibold mb-2">Rows</h2>
              <p className="text-3xl font-bold">{fileData.rows}</p>
            </CardBody>
          </Card>
          <Card>
            <CardBody>
              <h2 className="text-xl font-semibold mb-2">Columns</h2>
              <p className="text-3xl font-bold">{fileData.columns}</p>
            </CardBody>
          </Card>
        </div>

        <Card className="mb-6">
          <CardBody>
            <Table aria-label="Preview data table">
              <TableHeader>
                {columns.map((column) => (
                  <TableColumn key={column}>{column}</TableColumn>
                ))}
              </TableHeader>
              <TableBody>
                {fileData.preview_data.map((row: any, index: number) => (
                  <TableRow key={index}>
                    {columns.map((column) => (
                      <TableCell key={column}>
                        {column === "issue_description" ||
                        column === "resolution_notes" ? (
                          <Tooltip content={row[column]}>
                            {truncateText(row[column], MAX_TEXT_LENGTH)}
                          </Tooltip>
                        ) : (
                          row[column]
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
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
