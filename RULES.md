# Development Rules

## Core Rules

* Use FastAPI (NOT Flask)
* Keep ML logic separate from API routes
* Use modular and clean architecture
* Do NOT write all logic in one file
* Avoid hacks or quick fixes

---

## Backend Rules

* Separate:

  * routes (API endpoints)
  * services (business logic)
  * models (ML files)
* Use proper error handling
* Validate API inputs

---

## ML Rules

* Load model once (do NOT reload per request)
* Keep preprocessing separate
* Use scalable structure

---

## Frontend Rules

* Use Next.js (React)
* Use Tailwind CSS
* Create reusable components
* Focus on clean UI/UX

---

## General Rules

* Write readable code
* Use proper naming conventions
* Add comments where necessary
* Keep code production-ready

---

## Strict Warning

Do NOT:

* Mix frontend + backend logic
* Use Flask templates
* Write unstructured code
