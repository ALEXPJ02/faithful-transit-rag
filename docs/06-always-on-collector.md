# The Always-On Collector (GCP e2-micro)

> GitHub Actions admits the scheduled collector roughly once every two hours
> against a five-minute request. This is the machine that does not depend on
> anyone's scheduler, or anyone's laptop staying awake.

## Why not GitHub Actions alone

Measured on the first day of collection: 3 scheduled runs, gaps of 133 and 115
minutes. Every run succeeded — GitHub simply does not admit the job at the
requested rate. `--burst` (5 polls per admission) recovers some of it, but the
samples arrive in clumps hours apart, which is awkward for a time-based split and
leaves most of the day unobserved.

Actions stays enabled as a **backup**. Snapshot filenames are unique and the push
step rebases, so both writers can coexist without conflict.

## Why not your laptop

`caffeinate` blocks idle sleep but not lid-close sleep; `sudo pmset -a disablesleep 1`
does, at the cost of running a closed machine with no airflow for weeks. And a
laptop travels — the moment it leaves wifi, collection stops silently. Fine for a
few days on a desk, wrong for the collection window.

## Why e2-micro specifically

| Option | Reality |
| --- | --- |
| **GCP e2-micro** | Genuinely always-free, no expiry, no idle reclamation. 3 US regions, 30 GB disk. |
| Oracle Ampere A1 | Halved to 2 OCPU / 12 GB in June 2026. **Reclaims instances** idle below 20% CPU *and* network *and* memory over 7 days — exactly this workload's profile. |
| AWS t3.micro | 12 months only, which would actually suffice for a 6-week window. |

Oracle is the trap: a poller that wakes every two minutes, makes one HTTP call and
sleeps ticks every reclamation box, and losing the instance mid-window loses data
that cannot be re-collected. US-region latency is irrelevant at a two-minute cadence.

## Setup

**You need:** a Google account with billing enabled on a project. Billing is
required even for always-free resources; e2-micro within the free limits is not
charged. Set a budget alert if you want a hard signal.

### 1. Install and authenticate gcloud (on your Mac)

```bash
brew install --cask google-cloud-sdk
gcloud auth login
```

Then point it at a project and billing account. These read your real values
rather than asking you to substitute them — a literal `YOUR_PROJECT_ID` pasted
into a shell is accepted without complaint and fails confusingly later:

```bash
PROJECT_ID=$(gcloud projects list --format='value(projectId)' --limit=1)
BILLING_ID=$(gcloud billing accounts list --format='value(name)' --limit=1 | sed 's|.*/||')
echo "project=$PROJECT_ID billing=$BILLING_ID"
```

```bash
gcloud config set project "$PROJECT_ID"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ID"
gcloud services enable compute.googleapis.com
```

With more than one project or billing account, set those two variables by hand
instead of taking the first of each.

### 2. Create the instance

One command. The startup script provisions everything on first boot — no SSH.

Run this from the repository root — `--metadata-from-file` takes a relative
path, and the key is read straight out of your `.env`, so it never has to be
pasted anywhere:

```bash
TFNSW_KEY=$(grep '^TFNSW_API_KEY=' .env | cut -d= -f2-)
[ -n "$TFNSW_KEY" ] && echo "key found (${#TFNSW_KEY} chars)" || echo "NO KEY IN .env"
```

```bash
gcloud compute instances create transit-collector \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard \
  --metadata-from-file=startup-script=deploy/gcp-startup.sh \
  --metadata=tfnsw-api-key="$TFNSW_KEY"
```

**Stay inside the free tier:** the machine type must be `e2-micro`, the zone must be
in `us-west1`, `us-central1` or `us-east1`, and the disk must be `pd-standard` at
30 GB or less. Anything else is billable.

### 3. Confirm it is collecting

Give it two or three minutes to install, then:

```bash
gcloud compute ssh transit-collector --zone=us-central1-a \
  --command='systemctl is-active transit-poller && sudo journalctl -u transit-poller -n 20 --no-pager'
```

Healthy output shows `active` and repeating `poll ok — N entities seen, M observations written`.

## Getting the data back

The instance writes to SQLite at `/opt/transit-rag/data/delay_observations.db`,
which **deduplicates**: one row per stop event holding the latest value, rather
than one row per poll. That is the training shape, and it is why the always-on
box uses SQLite where the stateless Action uses CSV snapshots.

Pull a copy whenever you want one:

```bash
gcloud compute scp transit-collector:/opt/transit-rag/data/delay_observations.db \
  ./data/ --zone=us-central1-a
transit-poller --status
```

Do this at least weekly. The instance is the only copy of the deduplicated table —
the GitHub Actions backup holds raw CSV snapshots, which overlap but are not the
same thing. Losing the instance without a recent copy loses the window.

## Checking on it

```bash
gcloud compute instances list
gcloud compute ssh transit-collector --zone=us-central1-a \
  --command='sudo journalctl -u transit-poller --since "1 hour ago" | tail -30'
```

The unit is `Restart=always` with `StartLimitIntervalSec=0`, so a crash loop can
never leave systemd in a stopped state — on a window that cannot be re-run, a
retrying collector beats a cleanly-failed one.

The startup script is idempotent and re-runs on every boot, so a reboot re-clones
the latest `main`, reinstalls, and restarts. To pick up new code, just reset the
instance:

```bash
gcloud compute instances reset transit-collector --zone=us-central1-a
```

## When collection ends

```bash
gcloud compute scp transit-collector:/opt/transit-rag/data/delay_observations.db ./data/ --zone=us-central1-a
gcloud compute instances delete transit-collector --zone=us-central1-a
```

Pull the database **before** deleting. There is no other copy of it.
