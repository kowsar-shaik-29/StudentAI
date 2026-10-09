# STUDENT PROJECT: Student Guidance Platform

## Purpose
The Student Guidance Platform provides automated career counseling and academic study recommendations for engineering students.

## Architecture & System Design
- Frontend: Single-page application built with React and TypeScript.
- Backend: RESTful API built using FastAPI with asynchronous request handlers.
- Database: PostgreSQL is used as the primary relational database to ensure ACID compliance for student performance records.
- Authentication: Secure JWT (JSON Web Tokens) with refresh tokens.
- Vector Engine: Qdrant vector database for semantic retrieval of study notes.

## Documented Features
1. User registration and JWT-based authentication.
2. Academic study schedule tracking.
3. Automated resume-to-job matching via deterministic algorithms.

## Documented Limitations & Tradeoffs
1. PostgreSQL storage does not currently partition student tables across multiple regions.
2. Does not support automated SMS alerts (email only).
