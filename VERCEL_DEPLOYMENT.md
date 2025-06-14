# Vercel Deployment Guide

This guide explains how to deploy your Django AI Notebook application to Vercel.

## Prerequisites

1. Install Vercel CLI globally:
   ```bash
   npm install -g vercel
   ```

2. Make sure you have a Vercel account at https://vercel.com

## Environment Variables

Before deploying, set up these environment variables in your Vercel project settings:

### Required Variables:
- `DJANGO_SECRET_KEY`: A secure secret key for Django (generate a new one for production)
- `DEBUG`: Set to `false` for production
- `TOGETHER_API_KEY`: Your Together AI API key (if using Together AI)

### Optional Variables:
- `DATABASE_URL`: If using PostgreSQL instead of SQLite

## Deployment Steps

1. **Login to Vercel:**
   ```bash
   vercel login
   ```

2. **Deploy from your project directory:**
   ```bash
   vercel --prod
   ```
   
   OR
   
   **Deploy using Vercel dashboard:**
   - Go to https://vercel.com/dashboard
   - Click "New Project"
   - Import your GitHub repository
   - Configure environment variables
   - Deploy

3. **Set Environment Variables:**
   - In your Vercel dashboard, go to your project settings
   - Navigate to "Environment Variables"
   - Add the required variables listed above

## Important Notes

- **Database**: The current setup uses SQLite, which works for development but isn't ideal for production. Consider upgrading to PostgreSQL for production use.
- **Static Files**: Static files are automatically collected during build using `build_files.sh`
- **File Uploads**: Vercel has limitations on file uploads and storage. For production, consider using cloud storage (AWS S3, Google Cloud Storage, etc.)
- **Cold Starts**: Serverless functions on Vercel may have cold start delays, especially with ML models like sentence-transformers

## Production Recommendations

1. **Database**: Upgrade to PostgreSQL using Vercel Postgres or external providers
2. **File Storage**: Use cloud storage for uploaded documents
3. **Caching**: Implement Redis caching for better performance
4. **Environment Variables**: Never commit secrets to your repository

## Troubleshooting

- If deployment fails, check the build logs in Vercel dashboard
- Ensure all dependencies are listed in `requirements.txt`
- Make sure static files are being collected properly
- Check that environment variables are set correctly
