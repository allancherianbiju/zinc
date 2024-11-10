import React, { useState, useEffect } from "react";
import { Select, SelectItem, Button, Card, CardBody } from "@nextui-org/react";
import { useNavigate, useLocation } from "react-router-dom";
import Navbar from "../components/navbar";
import api from "../api";
import Fuse from "fuse.js";
import {
  IconClockHour4,
  IconFileDescription,
  IconCategory,
  IconUser,
  IconChartRadar,
  IconTablePlus,
} from "@tabler/icons-react";
import { toast } from "sonner";

const ColumnMapping = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const navigate = useNavigate();
  const location = useLocation();
  const { fileData } = location.state || {};
  const [columnNames, setColumnNames] = useState<string[]>([]);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [resolutionTimeField, setResolutionTimeField] =
    useState<string>("resolved_at");

  const incidentFields = [
    "Number",
    "Issue Description",
    "Reassignment Count",
    "Reopen Count",
    "Made SLA",
    "Caller ID",
    "Opened At",
    "Category",
    "Subcategory",
    "Symptom",
    "Confirmation Item",
    "Priority",
    "Assigned To",
    "Problem ID",
    "Resolved By",
    "Closed At",
    "Resolved At",
    "Resolution Notes",
  ];

  // Define field groups with their icons and fields
  const fieldGroups = {
    metrics: {
      icon: <IconChartRadar className="w-5 h-5" />,
      title: "Metric Fields",
      fields: {
        Number: {
          description: "Unique incident identifier",
        },
        "Reassignment Count": {
          description:
            "Number of times the incident has the group or support analysts changed",
        },
        "Reopen Count": {
          description:
            "Number of times the incident resolution was rejected by the caller",
        },
        "Made SLA": {
          description: "Indicates whether the incident exceeded the target SLA",
        },
        Priority: {
          description: "Calculated by the system based on impact and urgency",
        },
        "Problem ID": {
          description: "Identifier of the problem associated with the incident",
        },
      },
    },
    descriptive: {
      icon: <IconFileDescription className="w-5 h-5" />,
      title: "Descriptive Fields",
      fields: {
        "Issue Description": {
          description: "Description of the incident and related details",
        },
        "Resolution Notes": {
          description: "Notes describing how the incident was resolved",
        },
      },
    },
    classification: {
      icon: <IconCategory className="w-5 h-5" />,
      title: "Classification Fields",
      fields: {
        Category: {
          description: "First-level description of the affected service",
        },
        Subcategory: {
          description:
            "Second-level description of the affected service (related to category)",
        },
        Symptom: {
          description:
            "Description of the user perception about service availability",
        },
        "Confirmation Item": {
          description: "Identifier used to report the affected item",
        },
      },
    },
    time: {
      icon: <IconClockHour4 className="w-5 h-5" />,
      title: "Time Fields",
      fields: {
        "Opened At": {
          description: "Incident user opening date and time",
        },
        "Closed At": {
          description: "Incident user close date and time",
        },
        "Resolved At": {
          description: "Incident user resolution date and time",
        },
      },
    },

    assignment: {
      icon: <IconUser className="w-5 h-5" />,
      title: "Assignment Fields",
      fields: {
        "Caller ID": {
          description: "Identifier of the user affected by the incident",
        },
        "Assigned To": {
          description: "Identifier of the user in charge of the incident",
        },
        "Resolved By": {
          description: "Identifier of the user who resolved the incident",
        },
      },
    },

    other: {
      icon: <IconTablePlus className="w-5 h-5" />,
      title: "Other Fields",
      fields: {
        "Resolution Time Field": {
          description:
            "Select which timestamp to use for calculating incident resolution duration",
        },
      },
    },
  };

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

  // Add fuzzy matching logic
  useEffect(() => {
    if (columnNames.length > 0) {
      const fuse = new Fuse(columnNames, {
        threshold: 0.3,
        distance: 100,
      });

      // Create initial mapping based on fuzzy matching
      const initialMapping: Record<string, string> = {};
      let matchCount = 0;

      incidentFields.forEach((field) => {
        const results = fuse.search(field);
        if (results.length > 0) {
          initialMapping[field] = results[0].item;
          matchCount++;
        }
      });

      setMapping(initialMapping);

      // Show toast if any matches were found
      if (matchCount > 0) {
        toast.success(`${matchCount} fields were automatically mapped`, {
          description:
            "Please review and map the remaining fields manually before proceeding.",
        });
      }
    }
  }, [columnNames]);

  const handleMappingChange = (field: string, value: string) => {
    setMapping((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const requiredFields = [
    "Number",
    "Reassignment Count",
    "Priority",
    "Issue Description",
    "Resolution Notes",
    "Category",
    "Subcategory",
    "Symptom",
    "Opened At",
    "Closed At",
    "Resolved By",
  ];

  const handleSubmit = async () => {
    // Check if all required fields are mapped
    const missingFields = requiredFields.filter((field) => !mapping[field]);

    if (missingFields.length > 0) {
      toast.error("Missing required field mappings", {
        description: `Please map the following fields: ${missingFields.join(
          ", "
        )}`,
      });
      return;
    }

    try {
      await api.post("/mapping", {
        mapping,
      });
      navigate("/processing", {
        state: { mapping },
      });
    } catch (error) {
      console.error("Error submitting mapping:", error);
      toast.error("Failed to submit mapping", {
        description:
          "Please try again or contact support if the issue persists.",
      });
    }
  };

  console.log("Rendering with columnNames:", columnNames);

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-4">Column Mapping</h1>
        <p className="mb-6">
          Please map the columns from your uploaded file to the corresponding
          fields in our system.
        </p>

        {Object.entries(fieldGroups).map(([groupKey, group]) => (
          <Card key={groupKey} className="mb-6">
            <CardBody>
              <div className="flex items-center gap-2 mb-4">
                {group.icon}
                <h2 className="text-xl font-semibold">{group.title}</h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(group.fields).map(([field, fieldInfo]) => (
                  <Select
                    key={field}
                    label={field}
                    description={fieldInfo.description}
                    placeholder={`Select column for ${field}`}
                    className="max-w-xs"
                    selectedKeys={mapping[field] ? [mapping[field]] : []}
                    onChange={(e) => handleMappingChange(field, e.target.value)}
                    disabledKeys={Object.values(mapping).filter(
                      (value) => value !== mapping[field]
                    )}
                    isRequired={requiredFields.includes(field)}
                  >
                    {columnNames.map((column) => (
                      <SelectItem key={column} value={column}>
                        {column}
                      </SelectItem>
                    ))}
                  </Select>
                ))}
              </div>
            </CardBody>
          </Card>
        ))}

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
