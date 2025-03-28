# About the application

ZINC (Zero Incidents) is an application that aims to aid in creating a "zero incident" future by offering the following features:

- provide insights about the incidents raised and how they can be handled better in the future, with the goal of reducing the number of incidents raised to zero.
- predicting the estimated time and date for the final closure of a opened ticket based on the given attributes.

# Architecture

Frontend: React.js (Typescript)
UI Library: NextUI
Backend: FastAPI (Python)
Database: SQLite
GenAI Model: Llama 3.2 3B
GenAI Framework: Ollama + Langchain

# User Flow

1. User signs up and logs in
2. User uploads a CSV file
3. User is presented with a screen to review the data that they have uploaded.
4. User is then asked to review and edit the mapping of the columns of the dataset to the attributes of the model.
5. Upon submission, the data is processed and loaded into the database. An initial report is loaded into the cache.
6. The user is then redirected to the report page.
7. The report contains a few basic insights about the data along with GenAI recommendations generated from the data.

# About Dataset

Context
This is an event log of an incident management process extracted from data gathered from the audit system of an instance of the ServiceNowTM platform used by an IT company.

Content
The event log is enriched with data loaded from a relational database underlying a corresponding process-aware information system. Information was anonymized for privacy.

Number of instances: 141,712 events (24,918 incidents)
Number of attributes: 36 attributes (1 case identifier, 1 state identifier, 32 descriptive attributes, 2 dependent variables)

The attributed closed_at is used to determine the dependent variable for the time completion prediction task. The attribute resolved_at is highly correlated with closed_a. In this event log, some rows may have the same values (they are equal) since not all attributes involved in the real-world process are present in the log.

Attributes used to record textual information are not placed in this log. The missing values should be considered unknown information.

Acknowledgements
Thanks to University of SÃ£o Paulo, Brazil for this amazing dataset.

Inspiration
Create a Machine learning application which predicts the estimated time and date for the final closure of a opened ticket based on the given attributes.

# Attribute Information:

1. number: incident identifier (24,918 different values);
2. incident state: eight levels controlling the incident management process transitions from opening until closing the case;
3. active: boolean attribute that shows whether the record is active or closed/canceled;
4. reassignment_count: number of times the incident has the group or the support analysts changed;
5. reopen_count: number of times the incident resolution was rejected by the caller;
6. sys_mod_count: number of incident updates until that moment;
7. made_sla: boolean attribute that shows whether the incident exceeded the target SLA;
8. caller_id: identifier of the user affected;
9. opened_by: identifier of the user who reported the incident;
10. opened_at: incident user opening date and time;
11. sys_created_by: identifier of the user who registered the incident;
12. sys_created_at: incident system creation date and time;
13. sys_updated_by: identifier of the user who updated the incident and generated the current log record;
14. sys_updated_at: incident system update date and time;
15. contact_type: categorical attribute that shows by what means the incident was reported;
16. location: identifier of the location of the place affected;
17. category: first-level description of the affected service;
18. subcategory: second-level description of the affected service (related to the first level description, i.e., to category);
19. u_symptom: description of the user perception about service availability;
20. cmdb_ci: (confirmation item) identifier used to report the affected item (not mandatory);
21. impact: description of the impact caused by the incident (values: 1â€“High; 2â€“Medium; 3â€“Low);
22. urgency: description of the urgency informed by the user for the incident resolution (values: 1â€“High; 2â€“Medium; 3â€“Low);
23. priority: calculated by the system based on 'impact' and 'urgency';
24. assignment_group: identifier of the support group in charge of the incident;
25. assigned_to: identifier of the user in charge of the incident;
26. knowledge: boolean attribute that shows whether a knowledge base document was used to resolve the incident;
27. u_priority_confirmation: boolean attribute that shows whether the priority field has been double-checked;
28. notify: categorical attribute that shows whether notifications were generated for the incident;
29. problem_id: identifier of the problem associated with the incident;
30. rfc: (request for change) identifier of the change request associated with the incident;
31. vendor: identifier of the vendor in charge of the incident;
32. caused_by: identifier of the RFC responsible by the incident;
33. close_code: identifier of the resolution of the incident;
34. resolved_by: identifier of the user who resolved the incident;
35. resolved_at: incident user resolution date and time (dependent variable);
36. closed_at: incident user close date and time (dependent variable).
