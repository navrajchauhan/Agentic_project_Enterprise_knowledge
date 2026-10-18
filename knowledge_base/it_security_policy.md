# IT Security and Access Policy

*Document owner: Information Security Office. Effective date: 1 March 2025. Version: 3.1.*

## Purpose and Scope

This policy defines the mandatory security controls that protect company systems, customer data, and intellectual property. It applies to all employees, contractors, interns, and third-party personnel who access company networks, applications, or data from any device and any location. The Information Security Office owns the policy and reviews it annually or after a material incident. Non-compliance may result in immediate suspension of access and disciplinary action up to termination of employment or contract. Exceptions must be requested in writing, risk-assessed by the security team, and time-limited.

## Authentication and Credentials

### Passwords

All company accounts must use passwords with a minimum of 14 characters, combining upper- and lower-case letters, numbers, and symbols. Passwords must be rotated every 90 days, and the last 5 passwords may not be reused. Passwords must never be shared or written down, and an approved password manager is the only permitted place to store credentials. Accounts lock after five consecutive failed attempts and unlock through the service desk after identity verification. A password suspected to be compromised must be changed immediately and reported to the security team.

### Multi-Factor Authentication

Multi-factor authentication is mandatory for VPN, email, and all production systems, with no exception for convenience or seniority. Engineers and administrators with production access use phishing-resistant hardware keys. Where a hardware key is unavailable, an authenticator application with number matching is permitted. Push approvals must never be accepted unless the employee initiated the sign-in, and unexpected prompts must be reported as a suspected incident. Lost or replaced devices must be reported to the service desk the same day so the old factor is revoked.

## VPN and Remote Access

The company VPN must be used for all production access and split tunnelling is disabled, so all traffic from a connected device passes through the corporate egress and is inspected. VPN profiles are provisioned by IT and may not be modified. Only company-managed and compliant devices may connect to production networks; compliance means a running endpoint agent, full-disk encryption, and current operating system patches. Sessions time out after 30 minutes of inactivity, and public Wi-Fi may be used only with the VPN active. Connections from countries where the company has no legal presence require security approval before travel, and access from sanctioned countries is blocked.

## Endpoint Security and Encryption

Full-disk encryption (BitLocker or FileVault) is mandatory on all laptops. Every device must be enrolled in mobile device management, run the endpoint detection and response agent, lock the screen after 10 minutes, and install operating system updates within 14 days of release, or within 72 hours for critical patches. Local administrator rights are granted only where needed and reviewed every six months. USB storage devices are blocked by default; an exception requires written approval from the security team, is limited to a specific device and purpose, and expires after 90 days. Lost or stolen devices must be reported immediately so they can be remotely wiped.

## Access Control and Access Reviews

Access follows least privilege and is limited to the systems an employee needs for their role. Requests are raised through the service desk with manager approval, and production access additionally requires approval from the application owner. Access reviews happen quarterly, and privileged and administrator access is reviewed monthly; reviewers must confirm, modify, or revoke every entitlement within ten business days. Joiners receive baseline access on their start date, movers lose access to their previous role within five business days, and leavers are disabled within 24 hours of termination. Generic accounts are prohibited except for documented service accounts with a named owner, and orphaned accounts are removed within 30 days of detection.

## Data Classification and Handling

All company and customer data must be classified into one of four tiers: Public, Internal, Confidential, and Restricted. Public data may be shared freely. Internal data may be shared with employees and contracted partners under an agreement. Confidential data may be shared only on a need-to-know basis and must never be posted in public channels. Restricted data must be encrypted at rest and handled only through approved DLP tooling; it includes customer production data, credentials, personal data of EU residents, and financial records. Restricted data may not be copied to personal devices or pasted into external tools, including generative AI services.

| Tier | Examples | Handling requirement |
|---|---|---|
| Public | Marketing content, published documentation | No restrictions |
| Internal | Internal wiki, org charts, roadmaps | Employees and contracted partners |
| Confidential | Contracts, security reports, board papers | Need-to-know, approved storage |
| Restricted | Customer production data, credentials | Encrypted at rest, approved DLP tooling only |

## Software Installation and Acceptable Use

Software may only be installed from the approved software catalogue. Requests for new software are submitted through the service desk and undergo a security review covering licensing, vendor risk, and vulnerability history, completed within ten business days. Unapproved software found on a managed device is removed and the user is notified. Employees must not install personal licences, cracked software, or unapproved remote-access tools, and must not connect company devices to unsecured file-sharing services. Suspected shadow IT should be reported to the security team, which will assess whether the tool can be approved.

## Security Incidents and Reporting

Suspected security incidents must be reported to security@company.example within 1 hour of discovery. Reportable events include phishing messages, lost or stolen devices, suspected malware, accidental disclosure of Confidential or Restricted data, and any unexpected authentication prompt. Employees must not investigate on their own; they should preserve the device state and follow instructions from the security team. A Severity 1 (data breach) incident requires an initial response within 15 minutes, at which point the incident response team opens a bridge, assigns an incident lead, and starts the notification clock. Good-faith reporting is never punished, and delays caused by fear of blame are treated as more serious than the original mistake.

## Security Awareness Training

Security awareness training is completed annually by all employees, and new joiners complete the onboarding module within their first 30 days. The curriculum covers phishing, password hygiene, data handling, incident reporting, and secure remote work, and is updated when the threat landscape or policy changes. Completion is tracked by the security team, and employees who have not completed assigned training 30 days after the due date lose non-essential application access until it is complete. Simulated phishing campaigns run each quarter, and repeated failures trigger coaching rather than discipline.
