# GCP Architecture — VeritasRAG

Suggested GCP-based alternative architecture for VeritasRAG. This is a migration target from the current Render + Neon + Upstash deployment — no application code changes are required; only infrastructure and environment variables change.

---

## Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│  GCP  —  Alternative Architecture                                          │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  Cloud CDN  +  Cloud Load Balancing                                │   │
│  └──────────────────────────┬─────────────────────────────────────────┘   │
│                             │                                              │
│                             ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  Cloud Run  (Next.js)   — auto-scales to zero (cold starts at 0)   │   │
│  └──────────────────────────────────────────────────────────────────── ┘   │
│                                                                            │
│  ┌──────────────────────────┐   ┌────────────────────────────────────┐    │
│  │  Cloud Run               │   │  Cloud Run                         │    │
│  │  Django API              │──►│  FastAPI AI Service  (internal)    │    │
│  │  min-instances: 1        │   │  Vertex AI  Embeddings + Gemini    │    │
│  └──────────────────────────┘   └────────────────────────────────────┘    │
│                                                                            │
│  ┌──────────────────────────┐                                             │
│  │  Cloud Run Jobs          │   (replaces Celery + worker)                │
│  │  OR GKE Autopilot pod    │   trigger on Pub/Sub message                │
│  │  Celery worker           │                                             │
│  └──────────────────────────┘                                             │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Cloud SQL (PostgreSQL 16 or 17)  +  pgvector extension             │ │
│  │  HA configuration  •  read replica for analytics                    │ │
│  │  OR:  AlloyDB (PostgreSQL-compatible, better vector performance)     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Memorystore for Valkey or Redis (managed, Redis-compatible)         │ │
│  │  • Query cache, stats cache, Celery broker                          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Cloud Storage  (replaces Cloudinary)                                │ │
│  │  Signed upload URLs for browser-direct upload                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Vertex AI Vector Search  (optional scale-out)                       │ │
│  │  If pgvector hits limits: Vertex AI manages the HNSW index          │ │
│  │  Supports billions of vectors, fully managed                        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Secrets: Secret Manager  •  Networking: VPC + Direct VPC Egress           │
│  Observability: Cloud Logging + Cloud Trace                                │
│  CI/CD: GitHub Actions → Artifact Registry → Cloud Run deploy             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Service Mapping

| Need | Service |
|---|---|
| Container runtime | Cloud Run (serverless) or GKE Autopilot |
| Next.js frontend | Cloud Run or Firebase App Hosting (SSR-native, launched 2024) |
| PostgreSQL + pgvector | Cloud SQL for PostgreSQL 16 / 17 or AlloyDB |
| Vector search (at scale) | Vertex AI Vector Search |
| Redis cache + broker | Memorystore for Valkey (preferred) or Redis |
| File storage | Cloud Storage + signed upload URLs |
| LLM / embeddings | Vertex AI (Gemini + text-embedding) — same API, different endpoint |
| CDN | Cloud CDN + Cloud Load Balancing |
| Secrets | Secret Manager |
| Container registry | Artifact Registry |
| Async tasks | Cloud Run Jobs or Pub/Sub + Cloud Run trigger |

---

## Migration Notes

### From Cloudinary → Cloud Storage

Replace `CLOUDINARY_URL` with `GCS_BUCKET_NAME` + Application Default Credentials (Workload Identity on Cloud Run). Update `CloudinarySignatureView` in Django to generate GCS signed upload URLs via `google.cloud.storage.Blob.generate_signed_url`. FastAPI fetches files from GCS via `google.cloud.storage` instead of the Cloudinary SDK.

### From Neon → Cloud SQL / AlloyDB

`DATABASE_URL` swap only. Cloud SQL for PostgreSQL 16 supports pgvector via `CREATE EXTENSION vector`. AlloyDB is recommended for high-throughput vector workloads — it includes built-in vector similarity operators with better index performance than vanilla pgvector.

### From Upstash → Memorystore

`REDIS_URL` swap only. Memorystore is VPC-internal; connect via the internal IP.

**Valkey vs Redis:** Redis relicensed to SSPL in March 2024. GCP now offers **Memorystore for Valkey** (open-source fork, API-compatible with Redis 7.2) as the recommended choice for new deployments. Celery and Django cache backends work without any code changes.

**VPC connectivity:** Use **Direct VPC Egress** on Cloud Run (GA since 2024) to reach Memorystore — it is simpler and cheaper than the legacy VPC Connector (no additional instance required). Set `--vpc-egress=all-traffic` and specify the subnet on the Cloud Run service.

### From Render → Cloud Run

Each service maps to one Cloud Run service:

| Current | Cloud Run Service |
|---|---|
| `veritasrag-api` (Render Web Service) | Django API, `min-instances: 1`, public |
| `veritasrag-ai` (Render Web Service) | FastAPI AI service, VPC-internal (no public ingress) |
| `veritasrag-worker` (Render Background Worker) | Cloud Run Job triggered by Pub/Sub, or GKE Autopilot pod |

Pre-deploy migration hook becomes a Cloud Run Job running `python manage.py migrate` in the CI/CD pipeline before the new revision is promoted.

### Celery → Cloud Run Jobs + Pub/Sub (optional)

For a serverless-native approach, replace Celery entirely:
1. Django publishes a Pub/Sub message instead of `process_document.delay()`
2. A Pub/Sub push subscription triggers a Cloud Run Job per message
3. The Job runs the same FastAPI `/ingest` call

This eliminates the always-on worker cost and gives per-job scaling.

### Embeddings on Vertex AI

The current stack uses Google `gemini-embedding-001` at 768 dimensions for embeddings and OpenRouter for chat generation. On GCP, embeddings can move to the Vertex AI endpoint by setting `GOOGLE_GENAI_USE_VERTEXAI=1` and configuring `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`. Chat generation continues to use `OPENROUTER_API_KEY`.

### Vertex AI Vector Search (scale-out path)

When pgvector HNSW query latency exceeds SLA (typically > 50 million vectors):
1. Batch-export embeddings from PostgreSQL to Cloud Storage as JSON Lines
2. Create a Vertex AI Vector Search index from the export
3. Replace `retriever.py` ANN query with `aiplatform.MatchingEngineIndexEndpoint.match()`
4. Keep the BM25 + RRF merge — Cloud SQL FTS is unchanged

---

## Cost Estimate (minimal production)

| Service | Config | Est. monthly |
|---|---|---|
| Cloud Run (Django) | 1 vCPU / 512 MB, min-instances: 1 | ~$10 |
| Cloud Run (FastAPI) | 1 vCPU / 512 MB, min-instances: 1 | ~$10 |
| Cloud Run (Celery / Jobs) | per-use | ~$2 |
| Cloud SQL PostgreSQL | db-custom-1-3840, 20 GB SSD | ~$25 |
| Memorystore Redis | basic, 1 GB | ~$16 |
| Cloud Storage | 10 GB + operations | ~$1 |
| Cloud CDN + LB | 1 TB transfer | ~$10 |
| **Total** | | **~$74/mo** |

Cloud Run scales to zero outside business hours; actual cost for a demo workload will be significantly lower.
