<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="text"/>
<xsl:template match="/">
	<xsl:for-each select="valgrindoutput/error">
		<xsl:value-of select="kind"/>
		<xsl:text>: </xsl:text>
		<xsl:value-of select="xwhat/text"/>
		<xsl:for-each select="stack/frame">
			<xsl:text> at </xsl:text>
			<xsl:value-of select="dir"/>
			<xsl:text>/</xsl:text>
			<xsl:value-of select="file"/>
			<xsl:text>:</xsl:text>
			<xsl:value-of select="line"/>
			<xsl:text>&#10;</xsl:text>
		</xsl:for-each>
	</xsl:for-each>
</xsl:template>
</xsl:stylesheet>
