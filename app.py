from flask import Flask, render_template
from flask_cors import CORS


def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    from api import api_bp
    app.register_blueprint(api_bp)

    # Frontend routes
    @app.route('/')
    def index():
        """Home page showing all published routes"""
        return render_template('index.html')

    @app.route('/new')
    def new_route():
        """Route builder page"""
        return render_template('new.html')

    @app.route('/route/<route_id>')
    def view_route(route_id):
        """View a specific published route"""
        return render_template('route.html', route_id=route_id, route_name='Route')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
