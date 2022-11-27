# Answer Extractor Vietnamese Version

## Build

```
docker build --tag akagi2106/answer-extractor:v1 .
```

```
docker login
```

```
docker push akagi2106/answer-extractor:v1
```

```
docker run --name akagi2106 -p 8080:8080 akagi2106/answer-extractor:v1
```