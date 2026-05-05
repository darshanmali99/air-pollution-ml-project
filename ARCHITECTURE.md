# System Architecture – Air Pollution Forecasting System

## Overview

This project follows a 3-layer architecture:

Frontend (Next.js) → Backend API (FastAPI) → ML Engine (Models)

---

## High-Level Architecture

Frontend (UI Dashboard)

* Built using Next.js + Tailwind
* Displays AQI, predictions, charts

↓

Backend API (FastAPI)

* Handles all API requests
* Routes:

  * /predict
  * /historical-data
  * /metrics
  * /explain

↓

ML Layer

* Pretrained models (.pkl files)
* O3 and NO2 prediction
* SHAP explainability

---

## Backend Structure

backend/

* app/

  * main.py
  * routes/
  * services/
  * models/
  * utils/

---

## Data Flow

User Input → Frontend → API → Model → Prediction → UI

Dataset → Preprocessing → Model Training → Stored (.pkl)

---

## Deployment Plan

Frontend → Vercel
Backend → Render

---

## Goal

Transform this into a production-level AI system with:

* Clean architecture
* Scalable backend
* Modern UI dashboard
* Explainable ML outputs
