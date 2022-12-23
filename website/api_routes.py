from config import *
api_routes = Blueprint('api_routes', __name__, template_folder='templates')

# Retrieve email from JavaScript call and parse to Python register function as parameter
@api_routes.route("/api/internal/register", methods=["POST"])
def api_register_internal():
    email = request.json["email"]
    return jsonify(register(email))