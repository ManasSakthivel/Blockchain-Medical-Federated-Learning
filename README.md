# EHR Blockchain System - Flask Version

A secure, decentralized Electronic Health Record (EHR) system built with Flask, featuring blockchain integration and IPFS for decentralized storage.

## Features

- **Role-Based Access Control**: Admin, Doctor, and Patient roles with appropriate permissions
- **Blockchain Integration**: Medical records stored on Ethereum blockchain for immutability
- **IPFS Integration**: Large files and documents stored on IPFS for decentralized storage
- **Modern UI**: Beautiful, responsive interface built with Bootstrap 5
- **Secure Authentication**: Flask-Login with password hashing
- **Medical Records Management**: Create, view, and manage patient medical records
- **Consultation Booking**: Patients can book consultations with doctors
- **Real-time Blockchain Status**: Monitor blockchain connection status

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLAlchemy with SQLite (can be configured for PostgreSQL/MySQL)
- **Blockchain**: Ethereum (Ganache for development)
- **Storage**: IPFS for decentralized file storage
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Authentication**: Flask-Login
- **Smart Contracts**: Solidity

## Prerequisites

1. **Python 3.8+**
2. **Node.js** (for Ganache)
3. **Ganache** (local Ethereum blockchain)
4. **IPFS** (Kubo)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd flask-ehr-blockchain
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Ganache

1. Download and install [Ganache](https://www.trufflesuite.com/ganache)
2. Start Ganache and create a new workspace
3. Note the RPC URL (usually `http://127.0.0.1:7545`)

### 5. Set Up IPFS

1. Download and install [IPFS Kubo](https://dist.ipfs.tech/#go-ipfs)
2. Initialize IPFS: `ipfs init`
3. Start IPFS daemon: `ipfs daemon`
4. IPFS will be available at `http://127.0.0.1:5001`

### 6. Deploy Smart Contracts

1. Install Truffle globally:
```bash
npm install -g truffle
```

2. Compile and deploy contracts:
```bash
cd contracts
truffle compile
truffle migrate --network development
```

3. Note the deployed contract address and update it in the configuration.

### 7. Initialize Database

```bash
flask init-db
```

This will create the database and add an admin user:
- Email: `admin@ehr.com`
- Password: `admin123`

### 8. Run the Application

```bash
python run.py
```

The application will be available at `http://localhost:5000`

## Configuration

Create a `.env` file in the root directory:

```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///ehr.db
GANACHE_URL=http://127.0.0.1:7545
IPFS_URL=http://127.0.0.1:5001
CONTRACT_ADDRESS=your-deployed-contract-address
```

## Usage

### Admin Features

- **Dashboard**: Overview of doctors and patients
- **Manage Doctors**: Add and manage doctor accounts
- **Manage Patients**: View patient information
- **System Administration**: Monitor blockchain status

### Doctor Features

- **Dashboard**: Overview of consultations and patients
- **Patient Management**: View and manage patient records
- **Medical Records**: Create and update medical records
- **Consultations**: Manage patient consultations

### Patient Features

- **Dashboard**: Overview of medical records and consultations
- **Medical Records**: View personal medical records
- **Consultations**: Book and manage consultations with doctors
- **Profile Management**: Update personal information

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `GET /auth/logout` - User logout

### Admin Routes
- `GET /admin/dashboard` - Admin dashboard
- `GET /admin/doctors` - List all doctors
- `POST /admin/add-doctor` - Add new doctor
- `GET /admin/patients` - List all patients

### Doctor Routes
- `GET /doctor/dashboard` - Doctor dashboard
- `GET /doctor/patients` - List patients
- `GET /doctor/patient/<id>` - View patient details
- `POST /doctor/add-record/<patient_id>` - Add medical record
- `GET /doctor/consultations` - List consultations

### Patient Routes
- `GET /patient/dashboard` - Patient dashboard
- `GET /patient/records` - List medical records
- `GET /patient/record/<id>` - View medical record
- `GET /patient/consultations` - List consultations
- `POST /patient/book-consultation` - Book consultation

## Smart Contracts

### EHRContract.sol

Main contract for managing:
- Doctor registration and management
- Patient registration and management
- Medical record storage and retrieval
- Role-based access control

### Roles.sol

Library for managing role-based access control.

## Database Schema

### Users
- Basic user information and authentication
- Role-based access control

### Doctors
- Doctor-specific information
- Specialization and license details

### Patients
- Patient-specific information
- Medical history and contact details

### Medical Records
- Medical record data
- Blockchain and IPFS hashes
- Timestamps and validation status

### Consultations
- Consultation scheduling
- Status tracking

## Security Features

- **Password Hashing**: Secure password storage using Werkzeug
- **Session Management**: Flask-Login for secure sessions
- **Role-Based Access**: Different permissions for different user types
- **Blockchain Immutability**: Medical records stored on blockchain
- **IPFS Security**: Decentralized file storage with content addressing

## Development

### Project Structure

```
flask-ehr-blockchain/
├── app/
│   ├── __init__.py
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── static/
│   └── templates/
├── contracts/
├── migrations/
├── requirements.txt
├── run.py
└── README.md
```

### Adding New Features

1. Create new models in `app/models/`
2. Add routes in `app/routes/`
3. Create templates in `app/templates/`
4. Update navigation and permissions as needed

### Testing

```bash
python -m pytest tests/
```

## Deployment

### Production Setup

1. Use a production WSGI server (Gunicorn, uWSGI)
2. Set up a production database (PostgreSQL, MySQL)
3. Configure environment variables
4. Set up reverse proxy (Nginx)
5. Use HTTPS with SSL certificates

### Docker Deployment

```bash
docker build -t ehr-blockchain .
docker run -p 5000:5000 ehr-blockchain
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Contact: [your-email@example.com]

## Acknowledgments

- Original Angular version by [shamil-t](https://github.com/shamil-t/ehr-blockchain)
- Flask framework and community
- Ethereum and IPFS communities 