# Ethics, privacy and data provenance

## Position (recorded 2026-09-06)

The footage is supplied by the facility operator, who holds the consent and approval for its
collection. It is used for this research under that arrangement and is **not published,
redistributed, or shared outside the project**.

Consequences that follow from this, and that the thesis must honour:

1. **No dataset release.** The "documented multi-venue dataset" cannot be listed as a released
   artefact in the contributions. Either drop it, or release **derived features only** —
   embeddings, labels and the manifest, with no pixels — which is reproducible for anyone
   re-running the heads and carries no imagery. *(Resolves WP1-T5; decision still to be made
   between "drop" and "features-only".)*
2. **Figures.** Any frame reproduced in the thesis or slides has faces blurred, and identifiable
   venue branding removed or blurred where it is not needed to make the point.
3. **Retention.** Raw frames purged after 7 days in production; evidence images retained 365
   days for audit. Code enforces this (WP6-T8).
4. **The system never acts on its own.** Output is decision support. A human confirms every
   anomaly, and no automated financial action is taken (WP6-T12).

## Still worth confirming with the supervisor

The operator's approval covers *their* collection of the footage. Programmes often separately
require the student's *research use* to be signed off — usually a short form rather than a full
review. One question at the next supervision resolves it, and it is much cheaper to ask now than
to be asked at submission:

> The footage is provided by the facility operator, who holds consent for its collection. It is
> not published or shared, and I only report aggregate results with blurred figures. Does the
> programme still require a separate ethics form for my use of it?

## Workplace monitoring

The reconciliation layer compares booking records against observed occupancy, and can surface
discrepancy rates **per staff member**. That is employee monitoring, and it is a different
question from the footage consent.

Decision required (WP1-T4): either

- **(a)** report anomalies **per field only**, never aggregated per staff member — sufficient for
  the audit use case and side-steps the issue entirely; or
- **(b)** keep per-staff reporting, and document the legal basis and staff notification.

**Recommendation: (a).** The thesis contribution is that vision-based verdicts can be reconciled
against records at all; attributing discrepancies to individuals adds no scientific value and a
great deal of ethical exposure. The system can always be configured for (b) by an operator who
has their own basis for it.

## Scope limits (unchanged)

Scene state is classified, never identity. No face recognition, no re-identification, no tracking
of individuals between frames or across cameras. Tier-2 detection counts people and detects
attributes (hi-vis vest, machinery); it never matches a person to a person.
