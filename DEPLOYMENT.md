# Deployment Guide: Plant Disease Tool

## 🚀 Deployment Options

You have several deployment options. Choose the one that fits your needs:

### Option 1: **Streamlit Cloud** (Easiest, Recommended for beginners)
- ✅ Free tier available
- ✅ Automatic deployment from GitHub
- ✅ SSL/HTTPS included
- ✅ No server management
- ⚠️ Limited to Streamlit-compatible apps
- 📊 Compute: Limited CPU/memory on free tier

**Time to deploy:** 5 minutes

### Option 2: **Docker on VPS/Server** (Best for production)
- ✅ Full control
- ✅ Scalable
- ✅ Can handle large files
- ✅ Can add custom middleware
- ⏳ Requires server setup
- 💰 Requires paid hosting

**Time to deploy:** 20-30 minutes (depending on server setup)

### Option 3: **Heroku** (Deprecated, not recommended)
- ❌ Free tier discontinued (2022)
- 💰 Paid plans only ($7+/month)

### Option 4: **Self-Hosted on Local Machine**
- ✅ Simple to run
- ✅ Free
- ⚠️ Only works if machine stays on
- 📊 Limited by local network

**Time to deploy:** Already running! (just: `streamlit run app.py`)

---

## 🟢 **OPTION 1: Streamlit Cloud (Recommended for Quick Deployment)**

### Step 1: Enable Streamlit Cloud on Your GitHub Account

1. Go to [streamlit.io/app](https://streamlit.io/app)
2. Click "Deploy an app"
3. Click "I have a GitHub account"
4. Authorize Streamlit to access your GitHub
5. Select repository: `Jafor07/plant-disease-tool`
6. Select branch: `main`
7. Select main file path: `app.py`

### Step 2: Configure Environment (Optional)

Create `secrets.toml` in `.streamlit/` for sensitive data (if needed):

```bash
# .streamlit/secrets.toml
[database]
password = "your-secret-password"
```

### Step 3: Deploy

Click "Deploy!" in Streamlit Cloud dashboard.

**URL will be:** `https://share.streamlit.io/Jafor07/plant-disease-tool/main/app.py`

### Limitations of Streamlit Cloud
- Max upload: 200MB (⚠️ Your 2GB feature won't work)
- RAM: ~1GB
- Timeout: 15 minutes per session

**Note:** Use this for demos/testing only, not for large file handling.

---

## 🔵 **OPTION 2: Docker on VPS (Best for Production)**

### Step 1: Create Dockerfile

Already provided in `CONFIG_GUIDE.md`. If you don't have it:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY .streamlit/ .streamlit/

# Create data directories
RUN mkdir -p raw_images processed_images masks labels metadata

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Save as: `Dockerfile`

### Step 2: Create Docker Compose File

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    container_name: plant-disease-tool
    ports:
      - "8501:8501"
    volumes:
      - ./raw_images:/app/raw_images
      - ./processed_images:/app/processed_images
      - ./masks:/app/masks
      - ./labels:/app/labels
      - ./metadata:/app/metadata
      - ./.streamlit:/app/.streamlit
    environment:
      - STREAMLIT_CLIENT_MAXUPLOADSIZE=2000
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
```

Save as: `docker-compose.yml`

### Step 3: Build & Run Docker Image

```bash
# Build image
docker build -t plant-disease-tool:latest .

# Or use Docker Compose (easier)
docker-compose up -d

# Check if running
docker ps

# View logs
docker logs plant-disease-tool

# Stop the container
docker stop plant-disease-tool

# Restart
docker restart plant-disease-tool
```

### Step 4: Verify Deployment

```bash
# Access the app
curl http://localhost:8501

# Or open in browser
# http://localhost:8501
```

### Step 5: Deploy to VPS

**On your VPS/Server:**

```bash
# SSH into server
ssh user@your-server.com

# Clone repository
git clone https://github.com/Jafor07/plant-disease-tool.git
cd plant-disease-tool

# Install Docker (if not already installed)
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Make Dockerfile and docker-compose.yml executable
chmod +x Dockerfile

# Start the app
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f streamlit
```

### Step 6: Set Up Nginx Reverse Proxy (Optional but Recommended)

```bash
# Install Nginx
sudo apt-get install nginx

# Create Nginx config
sudo nano /etc/nginx/sites-available/plant-disease-tool
```

**Content:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    client_max_body_size 2000M;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Enable the config:**
```bash
sudo ln -s /etc/nginx/sites-available/plant-disease-tool /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 7: Set Up HTTPS with Let's Encrypt (Optional)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal (automatic, but verify)
sudo systemctl enable certbot.timer
```

---

## 🟡 **OPTION 3: AWS/Google Cloud Deployment**

### AWS EC2

**Step 1: Launch EC2 Instance**
- Instance type: `t3.medium` (1GB RAM minimum for large files)
- OS: Ubuntu 22.04 LTS
- Storage: 50GB+ SSD
- Security group: Allow port 80, 443

**Step 2: SSH into instance**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

**Step 3: Follow Docker deployment steps above**

**Step 4: Set up Route 53 DNS**
- Point domain to instance IP
- Set up health checks

### Google Cloud Run

```bash
# Build & push to Google Cloud
gcloud builds submit --tag gcr.io/PROJECT_ID/plant-disease-tool

# Deploy
gcloud run deploy plant-disease-tool \
  --image gcr.io/PROJECT_ID/plant-disease-tool:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 3600 \
  --set-env-vars STREAMLIT_CLIENT_MAXUPLOADSIZE=2000
```

---

## 📊 **Deployment Comparison**

| Feature | Streamlit Cloud | Docker + VPS | AWS EC2 | Google Cloud |
|---------|-----------------|--------------|---------|--------------|
| **Time to deploy** | 5 min | 20 min | 30 min | 20 min |
| **Cost** | Free (limited) | $5-20/month | $10-50/month | $10-50/month |
| **Large file support** | ❌ (200MB max) | ✅ (2GB) | ✅ (2GB) | ✅ (2GB) |
| **Memory** | ~1GB | 4-8GB configurable | 4GB+ | 4GB+ |
| **Customization** | ⚠️ Limited | ✅ Full | ✅ Full | ✅ Full |
| **Maintenance** | Minimal | Moderate | Moderate | Moderate |
| **Scaling** | Limited | ✅ Easy | ✅ Easy | ✅ Easiest |
| **Recommendation** | Demo/Testing | **Production** | Production | Production |

---

## 🛠️ **Recommended Setup (Best for You)**

Based on your needs (1GB+ files, batch processing), I recommend:

### **Docker on DigitalOcean (Simple & Affordable)**

**Why?**
- ✅ Simple deployment
- ✅ $4-6/month for entry-level
- ✅ Full 2GB upload support
- ✅ Easy scaling
- ✅ Great community support

**Steps:**
1. Create DigitalOcean account
2. Create Droplet (Ubuntu 22.04, $6/month)
3. SSH into droplet
4. Run 5 commands:
   ```bash
   git clone https://github.com/Jafor07/plant-disease-tool.git
   cd plant-disease-tool
   sudo apt-get update && sudo apt-get install docker.io docker-compose
   docker-compose up -d
   # Visit http://your-droplet-ip:8501
   ```
5. Optional: Add domain name (Route your domain to droplet IP)

---

## 📋 **Post-Deployment Checklist**

After deploying, verify:

- [ ] App loads at your URL (http or https)
- [ ] Can upload 100MB test file
- [ ] Can upload 1GB+ file (if using Docker)
- [ ] Annotation works correctly
- [ ] Can save and download files
- [ ] All output directories exist and have files
- [ ] No errors in server logs
- [ ] Health check passing (if configured)
- [ ] SSL/HTTPS working (if configured)
- [ ] File permissions correct (read/write access)

---

## 🔍 **Monitoring & Maintenance**

### Docker Logs
```bash
# View logs
docker-compose logs streamlit

# Follow logs in real-time
docker-compose logs -f streamlit

# Last 100 lines
docker-compose logs --tail 100 streamlit
```

### Docker Stats
```bash
# Monitor resource usage
docker stats plant-disease-tool

# Check disk usage
du -sh ./processed_images ./masks ./labels ./metadata

# Check disk space
df -h
```

### Restart & Updates
```bash
# Restart container
docker-compose restart streamlit

# Update code and restart
git pull
docker-compose down
docker-compose up -d

# View running containers
docker-compose ps

# Stop all services
docker-compose down
```

---

## 🚨 **Troubleshooting Deployment**

### Issue: "Connection refused" or app not accessible

```bash
# Check if container is running
docker-compose ps

# View error logs
docker-compose logs streamlit

# Check port is open
sudo lsof -i :8501

# Restart container
docker-compose restart streamlit
```

### Issue: Upload fails with large files

```bash
# Verify client max upload size in config
cat .streamlit/config.toml

# Should show: maxUploadSize = 2000

# Increase if needed, then restart
docker-compose restart streamlit
```

### Issue: Out of memory errors

```bash
# Check memory usage
docker stats plant-disease-tool

# Increase Docker memory (edit docker-compose.yml)
# Add under 'services.streamlit':
# mem_limit: 8g

# Or upgrade VPS to larger instance
```

### Issue: Nginx errors

```bash
# Test Nginx config
sudo nginx -t

# View Nginx logs
sudo tail -f /var/log/nginx/error.log

# Restart Nginx
sudo systemctl restart nginx
```

---

## 📚 **Deployment Documentation References**

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started)
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [DigitalOcean Tutorials](https://www.digitalocean.com/community/tutorials)
- [Let's Encrypt Setup](https://certbot.eff.org/instructions)

---

## 💡 **Next Steps**

1. **Choose deployment option** (Docker recommended for large files)
2. **Prepare environment:**
   - Domain name (optional)
   - VPS/Server (if not using Streamlit Cloud)
   - DNS settings
3. **Deploy using appropriate guide** (see above)
4. **Test thoroughly** before sharing publicly
5. **Monitor logs** for issues
6. **Set up auto-scaling** if handling many users

---

## ✅ **Quick Decision Tree**

```
Are you deploying for production?
│
├─ No (just testing) → Use: Streamlit Cloud
│
└─ Yes (real users, large files)
   │
   ├─ Have a VPS/Server? → Use: Docker + Nginx
   │
   └─ Need to buy hosting?
      │
      ├─ Budget < $10/month? → Get: DigitalOcean Droplet
      │
      ├─ Need more power? → Get: AWS EC2 t3.medium
      │
      └─ Want simplest? → Get: Google Cloud Run
```

---

## 📞 **Support for Deployment**

Need help with your specific setup? 

1. Check error logs: `docker-compose logs -f streamlit`
2. Run health checks: `curl http://localhost:8501/_stcore/health`
3. Test locally first: `streamlit run app.py`
4. Review CONFIG_GUIDE.md for your chosen platform
5. Check Streamlit docs for Streamlit-specific issues
6. Check Docker docs for Docker-specific issues
