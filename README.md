# img2txt Description Service

Private microservice for generating Russian clinical descriptions of
dermatoscopic images. The service receives an image and a PNG lesion mask from
`skin-cancer-ai`, waits for the classification result, generates text, and
returns it to the main backend by callback.

See [service/README.md](service/README.md) for API and Docker usage.
