"""
Script khởi động hệ thống nhận diện khuôn mặt
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app import app, initialize_models
from config import load_config


def setup_logging(config):
    """Setup logging configuration"""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/face_recognition_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress some noisy loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--host', type=str, default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--init-models', action='store_true', help='Initialize models only')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override config with command line arguments
    if args.host:
        config.web.host = args.host
    if args.port:
        config.web.port = args.port
    if args.debug:
        config.web.debug = True
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Face Recognition System")
    logger.info(f"Configuration: {config.web.host}:{config.web.port}")
    logger.info(f"Debug mode: {config.web.debug}")
    
    try:
        # Initialize models
        if initialize_models():
            logger.info("All models initialized successfully")
            
            if args.init_models:
                logger.info("Models initialized. Exiting.")
                return
            
            # Start web application
            app.run(
                host=config.web.host,
                port=config.web.port,
                debug=config.web.debug
            )
        else:
            logger.error("Failed to initialize models")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
