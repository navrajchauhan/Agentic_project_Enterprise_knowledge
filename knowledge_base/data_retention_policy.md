# Data Retention and Privacy Policy

*Document owner: Data Protection Office. Effective date: 1 February 2025. Version: 3.3.*

## Purpose and Scope

This policy defines how long the company keeps each category of data, where that data is stored, how it is protected, and how deletion requests are handled. It applies to all systems that store company or customer data, including production databases, analytics platforms, ticketing systems, backups, and collaboration tools. Each data owner is responsible for applying the retention periods to their systems and for maintaining a record of processing in the company data inventory. Data must not be kept longer than the periods below unless a legal hold or a statutory obligation applies. The data protection office reviews this policy annually with Legal, Finance, and Security, and privacy questions go to privacy@company.example.

## Support Ticket Retention

Support tickets are retained for 2 years after closure and then permanently deleted. The retention clock starts on the date the ticket is marked closed and applies to the ticket record, the message thread, attachments, and any chat transcripts linked to the ticket. Before deletion, tickets are reviewed for open complaints, legal holds, and audit requests; a ticket under any of those conditions is retained until the matter is closed. Aggregate statistics that contain no personal data may be retained indefinitely for capacity planning. Deletion is executed by an automated job that runs monthly, and each deletion run is logged and reviewed by the data protection office.

## Application and Security Logs

Application logs are retained for 90 days and are used for debugging, performance analysis, and product improvement. Security logs are retained for 13 months so that a full annual review cycle can be investigated, and include authentication events, privilege changes, and administrative actions. Logs are stored in an append-only system with access restricted to the security and reliability teams. Personal data in logs must be minimised at the point of collection, and sensitive fields must be masked. Logs may be extracted for an investigation and retained until the investigation closes. Both retention periods are enforced by lifecycle rules in the logging platform rather than by manual deletion.

## Backup Retention

Full backups run daily and are retained for 35 days, with incremental backups retained for the same period. Monthly archives are retained for 12 months and are stored in a separate region for disaster recovery. Backups are encrypted with the same standard as production data, and access requires break-glass approval from the data protection office and the reliability lead. Restore tests are performed quarterly and the results are recorded. When a deletion request is fulfilled, the record is removed from production immediately and from backups as each backup expires, so a residual copy may exist for up to 35 days, which remains within the 90-day completion window.

## Financial Records

Financial records are retained for 7 years to meet audit obligations. This includes invoices, purchase orders, expense reports, payroll journals, tax filings, and the correspondence that supports them. Records are stored in the finance system and in the document management platform with access limited to finance, legal, and internal audit. Records may not be destroyed while an audit, investigation, or tax enquiry is open, and the retention clock restarts when the matter closes. At the end of the retention period, records are deleted or anonymised in line with the deletion schedule, and the disposal is documented. Customer-requested deletion does not override a statutory financial retention obligation.

## Employee Records

Employee records are retained for 7 years after the employment end date. The record includes the contract, job history, performance reviews, compensation changes, leave balances, and statutory documents required by local law. Access is limited to the employee's manager during employment, People Operations, payroll, and legal. Medical information is kept separately and only for as long as the purpose requires. At the end of the retention period, records are destroyed securely and the destruction is recorded in the HR system. Employees may request a copy of their records during employment and for the retention period afterwards.

## Data Subject Rights and Deletion Requests

Individuals may request access, correction, or deletion of their personal data through the privacy portal or by contacting privacy@company.example. The company must acknowledge a data-subject deletion request within 30 days and complete it within 90 days. Identity is verified before any action is taken, and the request is logged with a reference number. Where a deletion cannot be completed because of a legal hold, a statutory retention obligation, or a pending dispute, the individual is told which data is retained and why. Deletion is propagated to processors and sub-processors, who must confirm completion in writing. Requests are tracked to closure by the data protection office.

## Data Residency

EU customer data is stored in the Frankfurt region, and India customer data is stored in the Mumbai region. Production storage, backups, and analytics for those customers stay in the contracted region, and access from outside the region is limited to authorised support staff under break-glass approval. Any transfer of personal data across regions requires an approved transfer mechanism such as standard contractual clauses and a documented transfer impact assessment. Customers may select the region at contract signature and may request a migration once per year, with a lead time of at least 60 days. Data residency controls are verified annually by the security team.

## Encryption Standards and Data-Protection Impact Assessment

Data at rest is encrypted with AES-256, and data in transit uses TLS 1.2 or higher. Encryption keys are managed in a central key management service, rotated annually, and access to keys is logged. Customer-managed keys are available on the Enterprise plan. A data-protection impact assessment is required before launching any feature that processes Restricted data, and it must be completed and approved by the data protection office before the feature enters production. The assessment covers purpose, data minimisation, retention, cross-border transfer, and security controls, and is reviewed again if the processing changes materially or at least every two years.
