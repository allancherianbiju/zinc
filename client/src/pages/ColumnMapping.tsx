import React, { useState, useEffect } from "react";
import {
  Select,
  SelectItem,
  Button,
  Breadcrumbs,
  BreadcrumbItem,
  Card,
  CardBody,
} from "@nextui-org/react";
import { useNavigate, useLocation } from "react-router-dom";
import Navbar from "../components/navbar";
import api from "../api";

const ColumnMapping = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { fileData } = location.state || {};
  const [columnNames, setColumnNames] = useState<string[]>([]);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [resolutionTimeField, setResolutionTimeField] =
    useState<string>("resolved_at");

  const incidentFields = [
    "number",
    "issue_description",
    "reassignment_count",
    "reopen_count",
    "made_sla",
    "caller_id",
    "opened_at",
    "category",
    "subcategory",
    "u_symptom",
    "cmdb_ci",
    "priority",
    "assigned_to",
    "problem_id",
    "resolved_by",
    "closed_at",
    "resolved_at",
    "resolution_notes",
  ];

  useEffect(() => {
    console.log("Full location state:", location.state);
    console.log("File Data:", fileData);

    if (!fileData) {
      console.warn("No file data available, redirecting to home");
      navigate("/");
      return;
    }

    let columns: string[] = [];
    if (Array.isArray(fileData.columns)) {
      columns = fileData.columns;
    } else if (
      typeof fileData.columns === "object" &&
      fileData.columns !== null
    ) {
      columns = Object.keys(fileData.columns);
    } else if (typeof fileData.columns === "number") {
      columns = Array.from(
        { length: fileData.columns },
        (_, i) => `Column ${i + 1}`
      );
    }

    console.log("Extracted columns:", columns);
    setColumnNames(columns);
  }, [fileData, navigate, location.state]);

  const handleMappingChange = (field: string, value: string) => {
    setMapping((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = async () => {
    try {
      await api.post("/mapping", {
        mapping,
        resolution_time_field: resolutionTimeField,
      });
      navigate("/processing", {
        state: { mapping, resolution_time_field: resolutionTimeField },
      });
    } catch (error) {
      console.error("Error submitting mapping:", error);
      // Handle error (e.g., show an error message to the user)
    }
  };

  console.log("Rendering with columnNames:", columnNames);

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <Breadcrumbs className="mb-6">
          <BreadcrumbItem>Home</BreadcrumbItem>
          <BreadcrumbItem>Upload</BreadcrumbItem>
          <BreadcrumbItem>Preview</BreadcrumbItem>
          <BreadcrumbItem>Map</BreadcrumbItem>
          <BreadcrumbItem isDisabled>Process</BreadcrumbItem>
          <BreadcrumbItem isDisabled>Report</BreadcrumbItem>
        </Breadcrumbs>

        <h1 className="text-2xl font-bold mb-4">Column Mapping</h1>
        <Card className="mb-6">
          <CardBody>
            <p>
              Please map the columns from your uploaded file to the
              corresponding fields in our system.
            </p>
          </CardBody>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {incidentFields.map((field) => (
            <Card key={field} className="mb-4">
              <CardBody>
                <Select
                  label={field}
                  placeholder={`Select column for ${field}`}
                  className="max-w-xs"
                  onChange={(e) => handleMappingChange(field, e.target.value)}
                  disabledKeys={Object.values(mapping).filter(
                    (value) => value !== mapping[field]
                  )}
                >
                  {columnNames.map((column) => (
                    <SelectItem key={column} value={column}>
                      {column}
                    </SelectItem>
                  ))}
                </Select>
              </CardBody>
            </Card>
          ))}
        </div>

        <Card className="mb-6">
          <CardBody>
            <Select
              label="Resolution Time Calculation"
              placeholder="Select field for resolution time"
              className="max-w-xs"
              onChange={(e) => setResolutionTimeField(e.target.value)}
            >
              <SelectItem key="resolved_at" value="resolved_at">
                resolved_at
              </SelectItem>
              <SelectItem key="closed_at" value="closed_at">
                closed_at
              </SelectItem>
            </Select>
          </CardBody>
        </Card>

        <div className="flex justify-end">
          <Button color="primary" onClick={handleSubmit}>
            Submit Mapping
          </Button>
        </div>
      </main>
    </div>
  );
};

export default ColumnMapping;
