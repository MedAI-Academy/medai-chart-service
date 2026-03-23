# MedAI Chart Service

Publication-quality scientific chart generation for PPTX embedding.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/charts/kaplan-meier` | POST | Kaplan-Meier survival curve |
| `/charts/forest-plot` | POST | Forest plot *(coming soon)* |
| `/charts/waterfall` | POST | Waterfall chart *(coming soon)* |
| `/charts/swimmer` | POST | Swimmer plot *(coming soon)* |

## Deploy to Railway

```bash
# 1. Create GitHub repo
cd medai-chart-service
git init
git add .
git commit -m "Initial: KM chart endpoint"
git remote add origin git@github.com:MedAI-Academy/medai-chart-service.git
git push -u origin main

# 2. In Railway Dashboard:
#    - New Project → Deploy from GitHub → select medai-chart-service
#    - Railway auto-detects Dockerfile
#    - Service will be available at: https://medai-chart-service-production-XXXX.up.railway.app
```

## Java Integration (Apache POI)

Your Java PPTX renderer calls this service to get chart PNGs:

```java
// In your PptxRenderService.java
private byte[] fetchChartPng(String endpoint, String jsonPayload) throws Exception {
    URL url = new URL(CHART_SERVICE_URL + endpoint);
    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
    conn.setRequestMethod("POST");
    conn.setRequestProperty("Content-Type", "application/json");
    conn.setDoOutput(true);
    
    try (OutputStream os = conn.getOutputStream()) {
        os.write(jsonPayload.getBytes(StandardCharsets.UTF_8));
    }
    
    try (InputStream is = conn.getInputStream()) {
        return is.readAllBytes();
    }
}

// Then embed into PPTX slide:
byte[] kmPng = fetchChartPng("/charts/kaplan-meier", kmJsonPayload);
XSLFPictureData picData = pptx.addPicture(kmPng, PictureData.PictureType.PNG);
XSLFSlide slide = pptx.getSlides().get(chartSlideIndex);
XSLFPictureShape pic = slide.createPicture(picData);
pic.setAnchor(new Rectangle2D.Double(LEFT, TOP, WIDTH, HEIGHT));
```

## KM Request Schema

```json
{
  "arms": [
    {
      "label": "Pembrolizumab + Chemo (n=200)",
      "times": [1.2, 3.4, 5.6, ...],
      "events": [1, 0, 1, ...],
      "color": "#2166AC"
    },
    {
      "label": "Placebo + Chemo (n=200)",
      "times": [0.8, 2.1, 4.5, ...],
      "events": [1, 1, 0, ...],
      "color": "#D6604D"
    }
  ],
  "title": "Overall Survival — Intent-to-Treat Population",
  "xlabel": "Time (months)",
  "ylabel": "Overall Survival (%)",
  "show_ci": true,
  "show_censoring": true,
  "show_at_risk": true,
  "show_median": true,
  "hr_text": "HR 0.56 (95% CI 0.44–0.71)\nStratified log-rank p < 0.001",
  "width": 10,
  "height": 7,
  "dpi": 300
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port (Railway sets this) |
| `CHART_SERVICE_URL` | — | Set in Java renderer env to point here |
