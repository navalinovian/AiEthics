from django.contrib import admin
from .models import Sentiment, Implement


class SentimentAdmin(admin.ModelAdmin):
    # define which columns displayed in changelist
    list_display = ('text', 'label')
    # add filtering by date
    list_filter = ('label',)
    # add search field 
    search_fields = ['text', 'label']

admin.site.register(Sentiment, SentimentAdmin)


admin.site.register(Implement, SentimentAdmin)