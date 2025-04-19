from django.urls import path
from . import views
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    # Authentication routes
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    
    # User account routes
    path('account/profile/', views.user_profile_view, name='user_profile'),
    path('account/settings/', views.settings_view, name='settings'),
    path('api/user/profile', views.update_user_profile, name='update_user_profile'),
    path('api/user/password', views.update_user_password, name='update_user_password'),
    
    # API endpoints for projects
    path('api/projects', views.get_projects, name='get_projects'),
    path('api/projects/create', csrf_exempt(views.create_project), name='create_project'),  # Added explicit create endpoint with CSRF exemption
    path('api/projects/<str:project_id>', views.get_project, name='get_project'),
    
    # HTML pages
    path('', views.index_view, name='index'),
    path('web.html', views.project_view, name='project_view'),  # Add this route for direct web.html access
    path('web/<str:project_id>/', views.project_view, name='project_view_with_id'),
    
    # API endpoints for document handling and QA 
    path('ask', views.ask_question, name='ask_question'),
    path('summarize', views.summarize_document, name='summarize_document'),
    path('reload/<str:project_id>', views.reload_document_index, name='reload_index'),
    path('check_file/<str:project_id>', views.check_file_exists, name='check_file'),
    path('upload_source/<str:project_id>', views.upload_source, name='upload_source'),
    path('rebuild_index/<str:project_id>', views.rebuild_document_index, name='rebuild_index'),
    path('list_sources/<str:project_id>', views.list_sources, name='list_sources'),
    path('delete_source/<str:project_id>/<str:filename>', views.delete_source, name='delete_source'),
    
    # New model-specific endpoints
    path('api/model/inference', views.model_inference, name='model_inference'),
    path('api/model/info', views.model_info, name='model_info'),
    path('api/model/health', views.model_health, name='model_health'),
]