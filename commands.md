# Google Cloud Platform Deployment Commands - Detailed Explanation

## 1. Artifact Registry Setup

### Create Docker Repository
```bash
gcloud artifacts repositories create iris-app-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Iris API"
```
**Purpose**: Creates a private Docker repository in Google Artifact Registry to store your container images.
- `iris-app-repo`: Name of your repository
- `--repository-format=docker`: Specifies this will store Docker images
- `--location=us-central1`: Geographic location for the repository
- Repository will be accessible at: `us-central1-docker.pkg.dev/YOUR_PROJECT/iris-app-repo`

### Configure Docker Authentication
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```
**Purpose**: One-time setup to authenticate Docker with Google Artifact Registry.
- Configures Docker to use your gcloud credentials when pushing/pulling images
- Adds credentials to your Docker configuration file (`~/.docker/config.json`)

## 2. Docker Image Management

### Tag Docker Image
```bash
docker tag iris-api "us-central1-docker.pkg.dev/${PROJECT_ID}/iris-app-repo/iris-api:v1"
```
**Purpose**: Creates a new tag for your local Docker image with the Artifact Registry path.
- `iris-api`: Your local image name
- `${PROJECT_ID}`: Environment variable containing your GCP project ID
- `iris-api:v1`: Image name and version tag in the registry

### Push Docker Image
```bash
docker push "us-central1-docker.pkg.dev/${PROJECT_ID}/iris-app-repo/iris-api:v1"
```
**Purpose**: Uploads your Docker image to Google Artifact Registry.
- Makes the image available for deployment on GKE
- Image is stored securely in your private repository

## 3. GKE Plugin Installation

### Install GKE Authentication Plugin
```bash
sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin
```
**Purpose**: Installs the authentication plugin required for kubectl to communicate with GKE clusters.
- Required for newer versions of kubectl (1.26+)
- Enables seamless authentication between kubectl and GKE

## 4. GKE Cluster Creation

### Create Autopilot Cluster
```bash
gcloud container clusters create-auto iris-cluster --region=us-central1
```
**Purpose**: Creates a fully-managed Kubernetes cluster using GKE Autopilot mode.
- `create-auto`: Uses Autopilot mode (Google manages nodes, scaling, security)
- `iris-cluster`: Name of your Kubernetes cluster
- `--region=us-central1`: Regional cluster (high availability across multiple zones)
- Autopilot handles: node provisioning, auto-scaling, security patches, and resource optimization

## 5. Project Configuration

### Set Project ID Variable
```bash
export PROJECT_ID=$(gcloud config get-value project)
echo "Your project ID is: $PROJECT_ID"
```
**Purpose**: Retrieves and stores your current GCP project ID in an environment variable.
- `gcloud config get-value project`: Gets the active project ID
- `export PROJECT_ID`: Makes it available for subsequent commands
- `echo`: Displays the project ID for verification

## 6. Service Account Setup

### Create Service Account
```bash
gcloud iam service-accounts create iris-api-sa \
    --display-name="Iris API Service Account"
```
**Purpose**: Creates a service account for your Iris API application.
- Service accounts provide identity for applications (not users)
- `iris-api-sa`: Service account name
- Full email: `iris-api-sa@${PROJECT_ID}.iam.gserviceaccount.com`

### Grant Storage Access
```bash
gcloud storage buckets add-iam-policy-binding gs://vertex-mlflow-artifacts-electric-wave-472614-d5 \
    --member="serviceAccount:iris-api-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```
**Purpose**: Grants the service account read access to your Cloud Storage bucket.
- Allows the API to read ML model artifacts from the bucket
- `roles/storage.objectViewer`: Read-only access to bucket objects
- Bucket contains: MLflow artifacts and trained models

## 7. Workload Identity Configuration

### Bind Service Accounts
```bash
gcloud iam service-accounts add-iam-policy-binding iris-api-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/iris-api-ksa]"
```
**Purpose**: Enables Workload Identity binding between Kubernetes and GCP service accounts.
- Workload Identity: Secure way for GKE pods to access GCP services
- `default/iris-api-ksa`: Kubernetes service account in the default namespace
- Allows pods using K8s SA to impersonate GCP SA

### Annotate Kubernetes Service Account
```bash
kubectl annotate serviceaccount iris-api-ksa \
    iam.gke.io/gcp-service-account=iris-api-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --overwrite
```
**Purpose**: Links the Kubernetes service account to the GCP service account.
- Annotation tells GKE which GCP service account to use
- `--overwrite`: Updates annotation if it already exists
- Completes the Workload Identity setup

## 8. Kubernetes Deployment

### Deploy Application
```bash
kubectl apply -f gke-deploy/deployment.yaml
```
**Purpose**: Creates or updates the Kubernetes Deployment for your Iris API.
- Deployment manages pods, replicas, rolling updates
- Typical contents: container image, resource requests/limits, environment variables, service account
- `-f`: Specifies the YAML file path

### Create Service
```bash
kubectl apply -f gke-deploy/service.yaml
```
**Purpose**: Creates or updates the Kubernetes Service to expose your API.
- Service provides stable networking endpoint for pods
- Types: ClusterIP (internal), LoadBalancer (external), NodePort
- Routes traffic to healthy pods

## 9. Monitoring Commands

### Watch Pod Status
```bash
kubectl get pods -w
```
**Purpose**: Monitors the status of all pods in real-time.
- `-w` (watch): Continuously updates as pod status changes
- Shows: pod name, ready status, restarts, age
- Press Ctrl+C to exit watch mode

### Watch Service Status
```bash
kubectl get service iris-api-service -w
```
**Purpose**: Monitors the service and waits for external IP assignment.
- For LoadBalancer services, shows the external IP when provisioned
- `-w`: Updates in real-time as service changes
- External IP is the public endpoint for your API

## Architecture Summary

**Flow of Deployment:**
1. Docker image built locally → pushed to Artifact Registry
2. GKE Autopilot cluster created in us-central1
3. Service account created with Storage access
4. Workload Identity configured for secure GCP access
5. Kubernetes Deployment pulls image and creates pods
6. Service exposes pods via LoadBalancer with external IP
7. Pods authenticate to GCP using Workload Identity
8. Application reads ML models from Cloud Storage bucket

**Security Features:**
- Private Docker registry (Artifact Registry)
- Service account with minimal permissions (least privilege)
- Workload Identity (no service account keys needed)
- GKE Autopilot security defaults (automatic updates, hardening)

**Key Resources Created:**
- Artifact Registry repository: `iris-app-repo`
- GKE cluster: `iris-cluster`
- GCP service account: `iris-api-sa`
- Kubernetes service account: `iris-api-ksa`
- Kubernetes deployment and service from YAML files