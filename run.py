"""Application entry point and operator CLI.

Run the development server:   python run.py         (needs SECRET_KEY or EHR_DEV_MODE=1)
Create an account:            flask --app run create-user --role admin --email you@example.org
                              (the password is prompted for, never taken from the source)

For anything other than local development use a production WSGI server, e.g.
``gunicorn 'run:app'``; this file never enables the Werkzeug debugger.
"""
import os

import click

from app import create_app, db
from app.access import password_problem, valid_email
from app.models import Doctor, Patient, User

app = create_app()

CLI_ROLES = ("admin", "doctor", "patient", "lab")


@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Doctor': Doctor, 'Patient': Patient}


@app.cli.command("create-user")
@click.option("--role", type=click.Choice(CLI_ROLES), required=True)
@click.option("--email", required=True)
@click.option("--username", default=None, help="Defaults to the part of the e-mail before '@'.")
@click.password_option(confirmation_prompt=True)
def create_user(role, email, username, password):
    """Create a user account (the only supported way to create an admin)."""
    email = email.strip().lower()
    username = (username or email.split("@")[0]).strip()
    if not valid_email(email):
        raise click.ClickException("Invalid e-mail address.")
    problem = password_problem(password)
    if problem:
        raise click.ClickException(problem)
    if User.query.filter((User.email == email) | (User.username == username)).first():
        raise click.ClickException("A user with that e-mail or username already exists.")
    user = User(username=username, email=email, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    click.echo(f"Created {role} account for {email} (id={user.id}).")


if __name__ == '__main__':
    # Development server only. Debug mode and public binding are opt-in.
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true"),
        host=os.environ.get("FLASK_RUN_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5002")),
    )
