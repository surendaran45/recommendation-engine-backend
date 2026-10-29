# Security checklist

- [ ] Rotate/revoke the TMDB API key previously committed in source history. Removing it from the current file does not revoke the old key or erase it from Git history.
- [ ] Keep `TMDB_API_KEY` in server-side environment settings only.
- [ ] Set `CORS_ORIGINS` to known frontend origins in production.
- [ ] Never load pickle artifacts from untrusted sources.
- [ ] Avoid exposing internal exceptions, local paths, provider details, or secrets in client responses.
- [ ] Do not treat the `/rate` demo endpoint as persistent storage.
- [ ] Add integration tests for model artifacts, missing users, empty ratings, invalid IDs, malformed data, and provider timeouts before production use.
