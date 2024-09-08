import { ColumnDef } from "@tanstack/react-table";

export const columns: ColumnDef<any>[] = [
  {
    accessorKey: "number",
    header: "Number",
  },
  {
    accessorKey: "issue_description",
    header: "Issue Description",
  },
  {
    accessorKey: "category",
    header: "Category",
  },
  {
    accessorKey: "priority",
    header: "Priority",
  },
  {
    accessorKey: "opened_at",
    header: "Opened At",
  },
  {
    accessorKey: "closed_at",
    header: "Closed At",
  },
];
