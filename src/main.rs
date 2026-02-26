// Rust Text Intelligence Starter - Backend Server
//
// Simple REST API server providing text intelligence analysis
// powered by Deepgram's Text Intelligence service.
//
// Key Features:
//   - Contract-compliant API endpoint: POST /api/text-intelligence
//   - Accepts text or URL in JSON body
//   - Supports multiple intelligence features: summarization, topics, sentiment, intents
//   - CORS-enabled for frontend communication
//   - JWT session auth with rate limiting (production only)

use axum::{
    extract::Query,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use chrono::Utc;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, process};
use tower_http::cors::{Any, CorsLayer};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Server configuration, overridable via environment variables.
struct Config {
    port: String,
    host: String,
}

/// Loads server configuration from environment variables with defaults.
fn load_config() -> Config {
    let port = env::var("PORT").unwrap_or_else(|_| "8081".to_string());
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    Config { port, host }
}

// ============================================================================
// SESSION AUTH - JWT tokens for production security
// ============================================================================

/// JWT claims structure for session tokens.
#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    iat: i64,
    exp: i64,
}

/// JWT token lifetime in seconds (1 hour).
const JWT_EXPIRY_SECS: i64 = 3600;

/// Initializes the session secret from environment or generates a random one.
fn init_session_secret() -> Vec<u8> {
    if let Ok(secret) = env::var("SESSION_SECRET") {
        if !secret.is_empty() {
            return secret.into_bytes();
        }
    }
    // Generate a random 32-byte secret for local development
    let mut secret = vec![0u8; 32];
    rand::thread_rng().fill_bytes(&mut secret);
    secret
}

/// Creates a signed HS256 JWT with the configured expiry duration.
fn create_jwt(secret: &[u8]) -> Result<String, String> {
    let now = Utc::now().timestamp();
    let claims = Claims {
        iat: now,
        exp: now + JWT_EXPIRY_SECS,
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret),
    )
    .map_err(|e| format!("Failed to create token: {}", e))
}

/// Validates a JWT signature and expiry. Returns Ok on success.
fn verify_jwt(token: &str, secret: &[u8]) -> Result<(), String> {
    let mut validation = Validation::default();
    validation.required_spec_claims.clear();
    validation.validate_exp = true;
    validation.validate_nbf = false;

    decode::<Claims>(token, &DecodingKey::from_secret(secret), &validation)
        .map(|_| ())
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("ExpiredSignature") {
                "token expired".to_string()
            } else {
                format!("invalid token: {}", msg)
            }
        })
}

/// Validates the Authorization header and returns an error response if invalid.
fn require_session(headers: &HeaderMap, secret: &[u8]) -> Option<(StatusCode, Json<serde_json::Value>)> {
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if auth_header.is_empty() || !auth_header.starts_with("Bearer ") {
        return Some((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "type": "AuthenticationError",
                    "code": "MISSING_TOKEN",
                    "message": "Authorization header with Bearer token is required"
                }
            })),
        ));
    }

    let token = &auth_header[7..];
    if let Err(e) = verify_jwt(token, secret) {
        let message = if e.contains("expired") {
            "Session expired, please refresh the page"
        } else {
            "Invalid session token"
        };
        return Some((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": message
                }
            })),
        ));
    }

    None
}

// ============================================================================
// API KEY LOADING
// ============================================================================

/// Reads the Deepgram API key from the environment. Exits if not set.
fn load_api_key() -> String {
    match env::var("DEEPGRAM_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            eprintln!("\n\u{274c} ERROR: Deepgram API key not found!\n");
            eprintln!("Please set your API key in .env file:");
            eprintln!("   DEEPGRAM_API_KEY=your_api_key_here\n");
            eprintln!("Get your API key at: https://console.deepgram.com\n");
            process::exit(1);
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Builds a structured error JSON response.
fn error_response(
    status: StatusCode,
    err_type: &str,
    code: &str,
    message: &str,
) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "type": err_type,
                "code": code,
                "message": message,
                "details": {}
            }
        })),
    )
}

// ============================================================================
// TOML METADATA
// ============================================================================

/// Represents the parsed deepgram.toml file for the [meta] section.
#[derive(Deserialize)]
struct DeepgramToml {
    meta: Option<toml::Value>,
}

// ============================================================================
// REQUEST / RESPONSE TYPES
// ============================================================================

/// JSON body for POST /api/text-intelligence.
#[derive(Deserialize)]
struct TextIntelligenceRequest {
    text: Option<String>,
    url: Option<String>,
}

/// Query parameters for POST /api/text-intelligence.
#[derive(Deserialize)]
struct TextIntelligenceParams {
    summarize: Option<String>,
    topics: Option<String>,
    sentiment: Option<String>,
    intents: Option<String>,
    language: Option<String>,
}

// ============================================================================
// SHARED APPLICATION STATE
// ============================================================================

/// Shared state passed to all route handlers.
#[derive(Clone)]
struct AppState {
    api_key: String,
    session_secret: Vec<u8>,
}

// ============================================================================
// ROUTE HANDLERS
// ============================================================================

/// Issues a signed JWT session token.
/// GET /api/session
async fn handle_session(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> impl IntoResponse {
    match create_jwt(&state.session_secret) {
        Ok(token) => (
            StatusCode::OK,
            Json(serde_json::json!({ "token": token })),
        ),
        Err(_) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "processing_error",
            "TOKEN_ERROR",
            "Failed to create session token",
        ),
    }
}

/// Processes text analysis requests via Deepgram Read API.
/// POST /api/text-intelligence
async fn handle_text_intelligence(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: HeaderMap,
    Query(params): Query<TextIntelligenceParams>,
    Json(body): Json<TextIntelligenceRequest>,
) -> impl IntoResponse {
    // Auth check
    if let Some(err) = require_session(&headers, &state.session_secret) {
        return err;
    }

    // Validate: exactly one of text or url
    let has_text = body.text.as_ref().map_or(false, |t| !t.is_empty());
    let has_url = body.url.as_ref().map_or(false, |u| !u.is_empty());

    if !has_text && !has_url {
        return error_response(
            StatusCode::BAD_REQUEST,
            "validation_error",
            "INVALID_TEXT",
            "Request must contain either 'text' or 'url' field",
        );
    }
    if has_text && has_url {
        return error_response(
            StatusCode::BAD_REQUEST,
            "validation_error",
            "INVALID_TEXT",
            "Request must contain either 'text' or 'url', not both",
        );
    }

    // If URL provided, fetch the text content from it
    let text_content = if has_url {
        let url_str = body.url.as_ref().unwrap();

        // Validate URL format
        if url::Url::parse(url_str).is_err() {
            return error_response(
                StatusCode::BAD_REQUEST,
                "validation_error",
                "INVALID_URL",
                "Invalid URL format",
            );
        }

        let client = reqwest::Client::new();
        match client.get(url_str).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    return error_response(
                        StatusCode::BAD_REQUEST,
                        "validation_error",
                        "INVALID_URL",
                        &format!("Failed to fetch URL: {}", resp.status()),
                    );
                }
                match resp.text().await {
                    Ok(text) => text,
                    Err(e) => {
                        return error_response(
                            StatusCode::BAD_REQUEST,
                            "validation_error",
                            "INVALID_URL",
                            &format!("Failed to read URL content: {}", e),
                        );
                    }
                }
            }
            Err(e) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "validation_error",
                    "INVALID_URL",
                    &format!("Failed to fetch URL: {}", e),
                );
            }
        }
    } else {
        body.text.unwrap_or_default()
    };

    // Check for empty text
    if text_content.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "validation_error",
            "EMPTY_TEXT",
            "Text content cannot be empty",
        );
    }

    // Extract query parameters for intelligence features
    let language = params.language.unwrap_or_else(|| "en".to_string());
    let summarize = params.summarize.unwrap_or_default();
    let topics = params.topics.unwrap_or_default();
    let sentiment = params.sentiment.unwrap_or_default();
    let intents = params.intents.unwrap_or_default();

    // Handle summarize v1 rejection
    if summarize == "v1" {
        return error_response(
            StatusCode::BAD_REQUEST,
            "validation_error",
            "INVALID_TEXT",
            "Summarization v1 is no longer supported. Please use v2 or true.",
        );
    }

    // Build Deepgram API URL with query parameters
    let mut dg_url = format!(
        "https://api.deepgram.com/v1/read?language={}",
        urlencoding::encode(&language)
    );

    if summarize == "true" || summarize == "v2" {
        dg_url.push_str("&summarize=v2");
    }
    if topics == "true" {
        dg_url.push_str("&topics=true");
    }
    if sentiment == "true" {
        dg_url.push_str("&sentiment=true");
    }
    if intents == "true" {
        dg_url.push_str("&intents=true");
    }

    // Build request body for Deepgram
    let dg_body = serde_json::json!({ "text": text_content });

    // Call Deepgram Read API
    let client = reqwest::Client::new();
    let dg_resp = match client
        .post(&dg_url)
        .header("Authorization", format!("Token {}", state.api_key))
        .header("Content-Type", "application/json")
        .json(&dg_body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("Deepgram API Error: {}", e);
            return error_response(
                StatusCode::BAD_REQUEST,
                "processing_error",
                "INVALID_TEXT",
                &format!("Failed to process text: {}", e),
            );
        }
    };

    let dg_status = dg_resp.status();
    let dg_body_text = match dg_resp.text().await {
        Ok(text) => text,
        Err(e) => {
            eprintln!("Deepgram Response Read Error: {}", e);
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "processing_error",
                "INVALID_TEXT",
                "Failed to read Deepgram response",
            );
        }
    };

    // Handle non-2xx from Deepgram
    if !dg_status.is_success() {
        eprintln!(
            "Deepgram API Error (status {}): {}",
            dg_status, dg_body_text
        );
        return error_response(
            StatusCode::BAD_REQUEST,
            "processing_error",
            "INVALID_TEXT",
            "Failed to process text",
        );
    }

    // Parse Deepgram response to extract results
    let dg_result: serde_json::Value = match serde_json::from_str(&dg_body_text) {
        Ok(val) => val,
        Err(e) => {
            eprintln!("Deepgram Response Parse Error: {}", e);
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "processing_error",
                "INVALID_TEXT",
                "Failed to parse Deepgram response",
            );
        }
    };

    // Return results (the Deepgram response includes a "results" key)
    let results = dg_result
        .get("results")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    (
        StatusCode::OK,
        Json(serde_json::json!({ "results": results })),
    )
}

/// Returns metadata from deepgram.toml.
/// GET /api/metadata
async fn handle_metadata() -> impl IntoResponse {
    let toml_content = match std::fs::read_to_string("deepgram.toml") {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading deepgram.toml: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "Failed to read metadata from deepgram.toml"
                })),
            );
        }
    };

    let cfg: DeepgramToml = match toml::from_str(&toml_content) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error parsing deepgram.toml: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "Failed to read metadata from deepgram.toml"
                })),
            );
        }
    };

    match cfg.meta {
        Some(meta) => {
            // Convert TOML value to JSON value
            let json_val = toml_value_to_json(&meta);
            (StatusCode::OK, Json(json_val))
        }
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Missing [meta] section in deepgram.toml"
            })),
        ),
    }
}

/// Converts a TOML value to a serde_json Value.
fn toml_value_to_json(val: &toml::Value) -> serde_json::Value {
    match val {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::json!(*i),
        toml::Value::Float(f) => serde_json::json!(*f),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(toml_value_to_json).collect())
        }
        toml::Value::Table(table) => {
            let mut map = serde_json::Map::new();
            for (k, v) in table {
                map.insert(k.clone(), toml_value_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    }
}

/// Returns a simple health check response.
/// GET /health
async fn handle_health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "service": "text-intelligence"
        })),
    )
}

/// Returns a 404 for unmatched routes.
async fn handle_not_found() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": "Not Found",
            "message": "Endpoint not found"
        })),
    )
}

// ============================================================================
// SERVER START
// ============================================================================

#[tokio::main]
async fn main() {
    // Load .env file (ignore error if not present)
    let _ = dotenvy::dotenv();

    // Load configuration
    let cfg = load_config();

    // Initialize session secret
    let session_secret = init_session_secret();

    // Load Deepgram API key
    let api_key = load_api_key();

    // Build shared application state
    let state = AppState {
        api_key,
        session_secret,
    };

    // Set up CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    // Set up routes
    let app = Router::new()
        .route("/api/session", get(handle_session))
        .route("/api/text-intelligence", post(handle_text_intelligence))
        .route("/api/metadata", get(handle_metadata))
        .route("/health", get(handle_health))
        .fallback(handle_not_found)
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port)
        .parse()
        .expect("Invalid address");

    println!();
    println!("{}", "=".repeat(70));
    println!("Backend API running at http://localhost:{}", cfg.port);
    println!();
    println!("GET  /api/session");
    println!("POST /api/text-intelligence (auth required)");
    println!("GET  /api/metadata");
    println!("GET  /health");
    println!("{}", "=".repeat(70));
    println!();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
