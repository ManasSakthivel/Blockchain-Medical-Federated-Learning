from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
import os

db = SQLAlchemy()  # Single source of db
login_manager = LoginManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///ehr.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    
    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    
    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User
        return User.query.get(int(user_id))
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.doctor import doctor_bp
    from app.routes.patient import patient_bp
    from app.routes.lab import lab_bp
    from app.routes.file_verification import file_verification_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(doctor_bp, url_prefix='/doctor')
    app.register_blueprint(patient_bp, url_prefix='/patient')
    app.register_blueprint(lab_bp, url_prefix='/lab')
    app.register_blueprint(file_verification_bp, url_prefix='/file-verification')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Add custom Jinja2 filters
    @app.template_filter('datetime')
    def datetime_filter(timestamp):
        """Convert Unix timestamp to readable datetime"""
        if timestamp:
            from datetime import datetime
            try:
                # Convert to datetime object
                dt = datetime.fromtimestamp(int(timestamp))
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                return str(timestamp)
        return 'N/A'
    
    return app 