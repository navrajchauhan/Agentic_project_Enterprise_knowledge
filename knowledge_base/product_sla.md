# Customer Support SLA and Escalation Matrix

*Document owner: Customer Support and Reliability Engineering. Effective date: 1 April 2025. Version: 5.0.*

## Purpose and Scope

This document defines the service levels the company commits to for its hosted platform, the severity model used to classify customer issues, and the escalation path that applies when a target is at risk. It applies to all customers on the Enterprise and Business plans and to every support engineer, incident commander, and engineering team that handles customer-reported issues. The SLA forms part of the master subscription agreement; where the agreement and this document conflict, the agreement prevails. Legacy plans follow the schedule attached to their order form. Support is delivered in English unless a local-language agreement is in place.

## Support Hours and Channels

Support is available 24x7 for P1 and P2 issues, and during business hours 09:00-18:00 IST for P3 and P4 issues. Business hours exclude public holidays in the support region. Customers raise issues through the support portal, by email to support@company.example, or by telephone for P1 and P2 using the number in the welcome pack. Every customer nominates up to five authorised contacts who may raise P1 issues. Tickets raised outside support hours are queued and acknowledged at the start of the next business day, except P1 and P2 tickets, which are paged immediately. The portal is the system of record, and verbal updates must be confirmed in the ticket.

## Severity Definitions

Every ticket is assigned a severity at intake using the definitions below. The company may re-classify a ticket where the reported impact does not match the definition, and will explain the change to the customer. Severity is based on business impact rather than on the number of users affected alone, and a single-tenant outage that blocks a customer's revenue operations can qualify as P1.

| Severity | Definition | Typical example |
|---|---|---|
| P1 | Critical outage; the service is unavailable or unusable | Login failures, data unavailable for all tenants |
| P2 | Major functionality impaired; a core workflow is broken | Reporting fails, integration sync stopped |
| P3 | Minor issue; a workaround exists | Slow page load, incorrect label |
| P4 | Cosmetic issue or general question | Copy change, how-to question |

Customers must include a reproducible description and the affected tenant identifiers in the ticket.

## Response and Resolution Targets

The following targets apply from the moment a ticket is accepted in the portal and are measured against the severity recorded at triage.

| Severity | First response | Resolution target | Coverage |
|---|---|---|---|
| P1 | 15 minutes | 4 hours | 24x7 |
| P2 | 1 hour | 8 business hours | 24x7 |
| P3 | 4 business hours | 3 business days | Business hours |
| P4 | 1 business day | Next scheduled release | Business hours |

For P1, first response means a human engineer has joined the bridge; an automated acknowledgement does not count. Resolution means the service is restored and confirmed by the customer, or a permanent fix is deployed. Time spent waiting on customer information is deducted from the resolution clock where the customer agrees.

## P1 Incident Management and Escalation

When a P1 is declared, an incident commander and the engineering on-call are paged immediately, and a dedicated incident channel is opened. The incident commander owns communication, coordination, and the decision log; the engineering on-call owns diagnosis and remediation. If the incident is not resolved within 2 hours, it escalates to the VP of Engineering, who assigns additional engineering resources and joins the customer bridge. Further escalation to the Chief Technology Officer occurs at four hours or on request from the customer's executive sponsor. The customer's named contact receives an update every 30 minutes while the incident is open. The incident commander must record all decisions, timestamps, and actions in the incident log.

## Uptime Commitment and Service Credits

The uptime commitment is 99.9% per calendar month, calculated as the total minutes in the month minus minutes of unavailability, divided by total minutes. Below 99.9% the customer receives a 10% monthly service credit, and below 99.0% the credit is 25%. To claim a credit, the customer must submit a request through the support portal within 30 days of the end of the affected month, with the affected dates and supporting evidence. Credits are applied to the next invoice and are not paid in cash. Scheduled maintenance announced at least 72 hours in advance is excluded from the calculation, as are outages caused by customer systems, third-party networks, or force majeure.

## Status Page and Customer Communications

The company maintains a public status page that shows current component status, active incidents, and scheduled maintenance. During a P1 the public status page must be updated every 30 minutes until the incident is resolved, and each update must state the affected components, the customer impact, and the next update time. Internal and customer-facing communications must use the same severity language and must not speculate about root cause before the post-incident review. A final update is published within two hours of resolution, confirming the restoration and stating when the post-incident review will be available.

## Post-Incident Reviews

Every P1 requires a written post-incident review within 5 business days, and P2 incidents receive a review at the engineering manager's discretion. The review follows a blameless format: a timeline of events, the customer impact, the contributing factors, and a set of corrective actions with owners and due dates. Corrective actions are tracked in the engineering issue tracker and reviewed at the weekly reliability meeting until they are closed. The customer receives the review and may request a walkthrough with the incident commander. Reviews are stored for three years and are used in the quarterly service review, where trends in incident causes and response times are presented to customers.

## Exclusions and Customer Responsibilities

The service levels do not apply to issues caused by customer-side configuration, unsupported integrations, network conditions outside the company's control, or use of the platform in breach of the agreement. Customers must maintain a supported browser, keep integration credentials current, and provide timely access to a technical contact during a P1. Scheduled maintenance is announced at least 72 hours in advance and is performed in the lowest-traffic window for the affected region. Beta features and preview functionality are excluded from the uptime commitment. Where an exclusion applies, the company will assist on a best-effort basis but will not issue a service credit.
