import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { DataTable } from "./components/data-table";
import { columns } from "./components/columns";
import {
  Breadcrumbs,
  BreadcrumbItem,
  Card,
  CardBody,
  Button,
} from "@nextui-org/react";
import Navbar from "./components/navbar";

export default function Preview() {
  const location = useLocation();
  const navigate = useNavigate();
  const { fileData } = location.state || {};

  if (!fileData) {
    navigate("/");
    return null;
  }

  console.log("Preview fileData:", fileData);

  // Ensure we're passing the correct structure to the mapping page
  const mappingData = {
    columns:
      fileData.preview_data && fileData.preview_data.length > 0
        ? Object.keys(fileData.preview_data[0])
        : Array.isArray(fileData.columns)
        ? fileData.columns
        : [],
    // Include any other necessary data
  };

  console.log("Mapping data to be passed:", mappingData);

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

        <DataTable columns={columns} data={fileData.preview_data} />

        <div className="mt-6 flex justify-end">
          <Button
            color="primary"
            onClick={() => {
              console.log("Navigating to mapping with data:", mappingData);
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
