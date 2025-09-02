from app import create_app, db
from app.models import User, Doctor, Patient
from werkzeug.security import generate_password_hash

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'User': User,
        'Doctor': Doctor,
        'Patient': Patient
    }

@app.cli.command()
def init_db():
    """Initialize the database with admin user."""
    with app.app_context():
        # Create admin user if it doesn't exist
        admin = User.query.filter_by(email='admin@ehr.com').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@ehr.com',
                password_hash=generate_password_hash('admin123'),
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print('Admin user created successfully!')
        else:
            print('Admin user already exists!')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 