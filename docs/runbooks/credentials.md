# Credentials: what needs one, where to get it, where to put it

Most of this pipeline needs nothing. Nine of the ten sources are open data with
no registration at all, and that is a deliberate property worth keeping — every
key is one more thing that expires while a backfill is running.

Run this before hunting through the docs; it reads the source configs and then
looks at what is actually on the machine:

```bash
meteopack credentials
```

**No secret ever goes in this repository, in a config file, or in a log.** A
config may name the *path* to a credential or the *environment variable* that
holds it, never the value. On a shared cluster, also keep secrets out of
`argv`: `curl -u user:pass` puts the password in the process list, where
everyone with `ps` can read it.

---

## 1. CEDA — needed now, expires every 3 days

**Used by:** `midas` (UK Met Office MIDAS-Open hourly surface observations).
**Nothing else needs it.**

CEDA tokens last about **72 hours and cannot be refreshed**, so this is a thing
you re-run, not a thing you set up once. `meteopack credentials` tells you how
old the current one is.

### The quick way

```bash
scripts/refresh-ceda-token.sh
```

It prompts for your CEDA username and password, POSTs them to CEDA's token API,
writes the token to `~/.ceda_token` with mode 600, and then verifies it against
the archive. The password is passed to `curl` on stdin and never stored.

The verification step matters more than it looks: `dap.ceda.ac.uk` answers
**HTTP 200 with an HTML login page** when it rejects a token, so a bad token
looks like a successful download until you read the bytes. The `midas` reader
checks for this too, which is why it fails with "CEDA returned a login page"
rather than a parse error.

### By hand

1. Log in and go to <https://services.ceda.ac.uk/account/token/>.
2. "Create new access token", enter your CEDA password, name it.
3. Copy the token into the file:
   ```bash
   umask 077 && cat > ~/.ceda_token    # paste, then Ctrl-D
   chmod 600 ~/.ceda_token
   ```

You may hold at most **2 active tokens** created this way.

### Forgotten which account it was

A CEDA token is a JWT, so the *expired* one still in `~/.ceda_token` says who
it belonged to. Decoding it needs no password and no network:

```bash
python3 -c 'import base64,json,sys
seg = open(sys.argv[1]).read().strip().split(".")[1]
d = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
print(d["preferred_username"], d["email"])' ~/.ceda_token
```

That prints the username the script asks for and the address to reset the
password with. (The payload of a JWT is base64, not encryption — which is also
why a token file is a secret worth `chmod 600`, expired or not.)

### Scripted, for Slurm

The API endpoint is `https://services.ceda.ac.uk/api/token/create/` (POST, HTTP
basic auth, returns JSON with an `access_token` key). This is the route CEDA
recommends for long-running jobs, because a job that outlives its token can
fetch another. `scripts/refresh-ceda-token.sh` uses it.

### Where meteopack looks

In order: `$CEDA_TOKEN_FILE`, then `extra.token_file` in
`configs/sources/midas.yaml`, then `~/.ceda_token`. The environment wins so one
job can be pointed at a fresh token without editing a shared config.

### If it still fails

A token only grants what your account already has. If
`https://dap.ceda.ac.uk/badc/ARCHIVE_INFO/ACCESS_TEST/RESTRICTED/TOKEN_CHECK`
rejects you, apply for dataset access through your CEDA account first.

---

## 2. EUMETSAT Data Store — needed to *build* the satellite archive

**Used by:** nothing in `meteopack`. The `eumetsat_seviri` source reads a local
archive of `.zip`/`.nat` scenes and never authenticates. The credential is what
`eumdac` needs to put the scenes there in the first place.

1. Create an EUMETSAT EO Portal account.
2. Go to <https://api.eumetsat.int/api-key/> (or your username menu → "API
   Key") and copy the **consumer key** and **consumer secret**.
3. Hand them to `eumdac` once:
   ```bash
   eumdac set-credentials <CONSUMER-KEY> <CONSUMER-SECRET>
   eumdac describe          # should list collections
   ```

`eumdac` stores them in `~/.eumdac/credentials`, mode 600, outside this repo.
The key and secret are long-lived; the *access token* eumdac derives from them
is short-lived and refreshed automatically.

The MSG15 RSS collection is `EO:EUM:DAT:MSG:MSG15-RSS`. The operational forward
path is MTG-I FCI, which uses the same account.

---

## 3. Keyless — nothing to do

| source | what it is |
|---|---|
| `dwd` | DWD open data, plain HTTPS directory tree |
| `belgium` | RMI Belgium open AWS feed |
| `dmi` | Danish MetObs — keyless since March 2026; the `api-key` parameter in the v2 docs is no longer required |
| `ea` | England & Wales Environment Agency, Open Government Licence |
| `meteofrance` | Météo-France public S3 climatological archive. Only their *near-real-time* API needs a token, and this is not it |
| `knmi_radar`, `knmi_insitu` | local archives delivered by KNMI; paths, not endpoints |
| `eumetsat_seviri` | local archive (see §2 for how it is filled) |

These are rate-limited rather than authenticated. The limits are in
`configs/sources/*.yaml` and are enforced in requests per second, not bytes —
KNMI in particular counts requests, which is why one-file-per-observation
sources are archived into tars on ingest.

---

## 4. Not wired up yet — what to request, and from whom

None of these are needed to run what exists today. They are the next sources,
listed here so the applications can go out before the code needs them. Several
take weeks.

| what | where | why it matters | lead time |
|---|---|---|---|
| **KNMI Data Platform** (`KNMI_API_KEY`) | <https://developer.dataplatform.knmi.nl/> | re-pulling the in-situ and radar archives from the API instead of the delivered tars | minutes, self-service |
| **EUMETNET OPERA** composites | via KNMI's OPERA membership | the pan-European radar leg | institutional, ask internally first |
| **Météo-France portail-api** | <https://portail-api.meteofrance.fr/> | near-real-time French observations; the archive leg needs no key | days |
| **MET Norway Frost** | <https://frost.met.no/auth/requestCredentials.html> | Norwegian archive; free client ID | minutes |
| **AEMET OpenData** | <https://opendata.aemet.es/centrodedescargas/altaUsuario> | Spain | minutes |
| **VMM waterinfo.be** | email for a client-credit code; `pywaterinfo` | the real Flemish rain-gauge density, 15-min | days |
| **SPW / WACONDAH** (Wallonia) | <https://hydrometrie.wallonie.be/> bulk request form | ~100 weighing gauges, ≈1 per 150 km² | days to weeks |
| **Met Office WOW / WOW-NL** | <https://mowowprod.portal.azure-api.net/> | the only free route to *historical* private-station data; one account covers NL and BE | days |
| **Netatmo** | <https://dev.netatmo.com/> | densest NL network (~1 per 9 km²) but the public API has **no history**, so polling must start before you need the data | days; bulk data is **not redistributable** |
| **CEDA EUMETNET-Netatmo sandbox** | application through your CEDA account | 31 countries, all of 2020, already quality-controlled | weeks, access-restricted |

One note that saves work:

- **Netatmo's public API returns only currently-operational stations.** There is
  no backfill. If you want 2027's data you have to be polling in 2026.

---

## 5. Keeping secrets off a shared filesystem

The project directory is group-readable, so **no credential belongs in it** —
not in the repo, not beside the data. Everything above lives in `$HOME` with
mode 600.

- `meteopack credentials` warns if a token file is readable by others, and so
  does the `midas` reader on every run.
- `.env` (from `.env.example`) holds *paths*, not secrets, and is gitignored
  anyway.
- For Slurm, export the variable in the job script from a file in `$HOME`;
  do not bake it into the submission or pass it on a command line.
