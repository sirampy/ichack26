from flask import Flask, render_template
from flask_cors import CORS


def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    from api import api_bp
    app.register_blueprint(api_bp)

    # Frontend route
    @app.route('/')
    def index():
        """Serve the frontend"""
        return render_template('index.html')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
