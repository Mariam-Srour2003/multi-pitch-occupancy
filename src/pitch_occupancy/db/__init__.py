"""SQLite schema and access.

Tables: fields, cameras, rental_slots, frame_samples, slot_evaluations, bookings,
reconciliations. Overrides are audit fields on slot_evaluations, and they double as
fresh ground-truth labels for periodic head re-fits (WP6-T7).
"""
